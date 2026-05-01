#!/usr/bin/env python3
"""Plot predicted vs GT action chunks from offline diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot chunk curves from diagnose_action_chunk outputs.")
    parser.add_argument(
        "--diag-dir",
        action="append",
        required=True,
        help="Diagnostic output directory containing diagnostics.json and chunks.npz. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label per --diag-dir. Defaults to directory name.",
    )
    parser.add_argument("--sample-id", type=int, default=0, help="Sample id to visualize.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("action_chunk_curve_compare.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_run(diag_dir: Path, sample_id: int) -> dict[str, object]:
    diag_dir = diag_dir.expanduser().resolve()
    json_path = diag_dir / "diagnostics.json"
    npz_path = diag_dir / "chunks.npz"
    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Missing diagnostics.json or chunks.npz in {diag_dir}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    row = next((item for item in rows if int(item["sample_id"]) == sample_id), None)
    if row is None:
        raise KeyError(f"sample_id={sample_id} not found in {json_path}")

    chunks = np.load(npz_path)
    pred_key = f"pred_{sample_id}"
    gt_key = f"gt_{sample_id}"
    if pred_key not in chunks.files or gt_key not in chunks.files:
        raise KeyError(f"{pred_key} or {gt_key} missing in {npz_path}")

    return {
        "dir": diag_dir,
        "row": row,
        "pred": np.asarray(chunks[pred_key], dtype=np.float32),
        "gt": np.asarray(chunks[gt_key], dtype=np.float32),
    }


def main() -> int:
    args = parse_args()
    diag_dirs = [Path(item) for item in args.diag_dir]
    labels = args.label or []
    if labels and len(labels) != len(diag_dirs):
        raise ValueError("--label count must match --diag-dir count.")
    if not labels:
        labels = [path.name for path in diag_dirs]

    runs = [load_run(path, args.sample_id) for path in diag_dirs]

    dim_groups = [
        ("EEF Pos", slice(0, 3), ["x", "y", "z"]),
        ("EEF Rot", slice(3, 6), ["rx", "ry", "rz"]),
        ("Hand 0-5", slice(6, 12), [f"h{i}" for i in range(6)]),
        ("Hand 6-11", slice(12, 18), [f"h{i}" for i in range(6, 12)]),
    ]

    sample_row = runs[0]["row"]
    args.output = args.output.expanduser().resolve()
    title = (
        f"Action Chunk Curves | sample_id={args.sample_id} "
        f"episode={sample_row['episode_index']} frame={sample_row['frame_idx']}"
    )
    if plt is not None:
        render_with_matplotlib(runs, labels, dim_groups, title, args.output)
    else:
        render_with_pil(runs, labels, dim_groups, title, args.output)
    print(f"Saved plot to {args.output}")
    return 0


def render_with_matplotlib(runs, labels, dim_groups, title: str, output: Path) -> None:
    fig, axes = plt.subplots(len(dim_groups), 1, figsize=(14, 14), sharex=True)
    chunk_steps = np.arange(runs[0]["pred"].shape[0])
    colors = ["#d1495b", "#00798c", "#edae49", "#30638e", "#8f2d56"]

    gt = runs[0]["gt"]
    for ax, (panel_title, dim_slice, names) in zip(axes, dim_groups):
        gt_block = gt[:, dim_slice]
        for idx in range(gt_block.shape[1]):
            ax.plot(
                chunk_steps,
                gt_block[:, idx],
                color="#111111",
                linewidth=2.2,
                alpha=0.85,
                label=f"GT {names[idx]}" if idx == 0 else None,
            )
        for run_idx, run in enumerate(runs):
            pred_block = run["pred"][:, dim_slice]
            color = colors[run_idx % len(colors)]
            for idx in range(pred_block.shape[1]):
                ax.plot(
                    chunk_steps,
                    pred_block[:, idx],
                    color=color,
                    linewidth=1.4,
                    alpha=0.9,
                    linestyle="--",
                    label=f"{labels[run_idx]} pred" if idx == 0 else None,
                )
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=min(4, len(runs) + 1), fontsize=9)

    fig.suptitle(title, fontsize=15)
    axes[-1].set_xlabel("Chunk Step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180)


def render_with_pil(runs, labels, dim_groups, title: str, output: Path) -> None:
    width = 1600
    panel_height = 300
    top_margin = 90
    left_margin = 90
    right_margin = 40
    bottom_margin = 50
    height = top_margin + panel_height * len(dim_groups) + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#d1495b", "#00798c", "#edae49", "#30638e", "#8f2d56"]
    gt_color = "#111111"
    plot_width = width - left_margin - right_margin
    chunk_steps = np.arange(runs[0]["pred"].shape[0], dtype=np.float32)

    draw.text((left_margin, 20), title, fill="black", font=font)
    for panel_idx, (panel_title, dim_slice, names) in enumerate(dim_groups):
        y0 = top_margin + panel_idx * panel_height
        y1 = y0 + panel_height - 40
        x0 = left_margin
        x1 = left_margin + plot_width
        draw.rectangle((x0, y0, x1, y1), outline="#bbbbbb", width=1)
        draw.text((x0, y0 - 18), panel_title, fill="black", font=font)

        series = []
        gt_block = runs[0]["gt"][:, dim_slice]
        for idx in range(gt_block.shape[1]):
            series.append((f"GT {names[idx]}", gt_color, gt_block[:, idx], 3))
        for run_idx, run in enumerate(runs):
            pred_block = run["pred"][:, dim_slice]
            color = colors[run_idx % len(colors)]
            for idx in range(pred_block.shape[1]):
                series.append((f"{labels[run_idx]} {names[idx]}", color, pred_block[:, idx], 2))

        all_values = np.concatenate([item[2] for item in series], axis=0)
        vmin = float(all_values.min())
        vmax = float(all_values.max())
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        pad = 0.08 * (vmax - vmin)
        vmin -= pad
        vmax += pad

        for grid_ratio in (0.25, 0.5, 0.75):
            gy = y0 + int((1.0 - grid_ratio) * (y1 - y0))
            draw.line((x0, gy, x1, gy), fill="#eeeeee", width=1)

        def map_xy(step_idx: int, value: float) -> tuple[int, int]:
            x = x0 + int(step_idx * plot_width / max(1, len(chunk_steps) - 1))
            norm = (value - vmin) / (vmax - vmin)
            y = y1 - int(norm * (y1 - y0))
            return x, y

        for _, color, values, line_width in series:
            points = [map_xy(i, float(v)) for i, v in enumerate(values)]
            draw.line(points, fill=color, width=line_width)

        draw.text((10, y0 + 2), f"{vmax:.3f}", fill="#555555", font=font)
        draw.text((10, y1 - 10), f"{vmin:.3f}", fill="#555555", font=font)

        legend_x = x1 - 260
        legend_y = y0 + 8
        legend_items = [("GT", gt_color)] + [(label, colors[i % len(colors)]) for i, label in enumerate(labels)]
        for item_idx, (label, color) in enumerate(legend_items):
            yy = legend_y + item_idx * 16
            draw.line((legend_x, yy + 6, legend_x + 20, yy + 6), fill=color, width=3)
            draw.text((legend_x + 26, yy), label, fill="black", font=font)

    image.save(output)


if __name__ == "__main__":
    raise SystemExit(main())
