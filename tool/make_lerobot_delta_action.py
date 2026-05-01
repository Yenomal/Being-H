#!/usr/bin/env python3
"""Create a LeRobot dataset variant with an added delta-action column."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy.spatial.transform import Rotation as R
except ImportError:
    np = None
    pd = None
    R = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a LeRobot dataset, convert absolute EEF actions into delta actions, "
            "and write a new dataset directory with an added action column."
        )
    )
    parser.add_argument("--input-dataset", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--output-dataset", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument(
        "--state-column",
        type=str,
        default="observation.state",
        help="Absolute state pose column. Expected [xyz, axis-angle/quat, hand...] layout.",
    )
    parser.add_argument(
        "--action-column",
        type=str,
        default="action",
        help="Absolute action pose column. Expected [xyz, axis-angle/quat, hand...] layout.",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="world_delta_action",
        help="Name of the generated delta action column.",
    )
    parser.add_argument(
        "--delta-frame",
        choices=["world", "local"],
        default="world",
        help="Interpret position/rotation delta in world or EEF-local frame.",
    )
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order when source vectors are 19D pose+hand arrays.",
    )
    parser.add_argument(
        "--hand-mode",
        choices=["copy", "delta"],
        default="copy",
        help="How to handle dims after the first 6 EEF dims.",
    )
    parser.add_argument(
        "--replace-action",
        action="store_true",
        help=(
            "Overwrite the original action column with delta action in the output dataset. "
            "Useful when you want downstream training code to keep reading `action` unchanged."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to carry unchanged files into the output dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output dataset first if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def copy_or_link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def copy_dataset_skeleton(src_root: Path, dst_root: Path, copy_mode: str) -> None:
    for path in sorted(src_root.rglob("*")):
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if (
            path.suffix == ".parquet"
            or rel.as_posix() == "meta/info.json"
            or rel.as_posix() == "meta/stats.json"
        ):
            continue
        copy_or_link_file(path, dst, copy_mode)


def quat_to_rot(quat: np.ndarray, order: str) -> R:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if order == "wxyz":
        quat = quat[[1, 2, 3, 0]]
    return R.from_quat(quat)


def split_pose(vector: np.ndarray, quat_order: str) -> tuple[np.ndarray, R, np.ndarray]:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape[0] == 18:
        pos = vec[:3]
        rot = R.from_rotvec(vec[3:6])
        tail = vec[6:]
        return pos, rot, tail
    if vec.shape[0] == 19:
        pos = vec[:3]
        rot = quat_to_rot(vec[3:7], quat_order)
        tail = vec[7:]
        return pos, rot, tail
    raise ValueError(f"Expected pose vector dim 18 or 19, got {vec.shape}")


def make_delta_action(
    state_vec: np.ndarray,
    action_vec: np.ndarray,
    delta_frame: str,
    hand_mode: str,
    quat_order: str,
) -> np.ndarray:
    state_pos, state_rot, state_tail = split_pose(state_vec, quat_order)
    action_pos, action_rot, action_tail = split_pose(action_vec, quat_order)

    if delta_frame == "world":
        delta_pos = action_pos - state_pos
        delta_rot = (action_rot * state_rot.inv()).as_rotvec()
    else:
        delta_pos = state_rot.inv().apply(action_pos - state_pos)
        delta_rot = (state_rot.inv() * action_rot).as_rotvec()

    if hand_mode == "delta":
        hand = action_tail - state_tail
    else:
        hand = action_tail

    return np.concatenate([delta_pos, delta_rot, hand], axis=0).astype(np.float32)


def transform_parquet(
    parquet_path: Path,
    dst_path: Path,
    state_column: str,
    action_column: str,
    output_column: str,
    replace_action: bool,
    delta_frame: str,
    hand_mode: str,
    quat_order: str,
) -> None:
    df = pd.read_parquet(parquet_path)
    if state_column not in df.columns:
        raise KeyError(f"{state_column} not found in {parquet_path}")
    if action_column not in df.columns:
        raise KeyError(f"{action_column} not found in {parquet_path}")

    deltas = [
        make_delta_action(
            np.asarray(state_val),
            np.asarray(action_val),
            delta_frame=delta_frame,
            hand_mode=hand_mode,
            quat_order=quat_order,
        )
        for state_val, action_val in zip(df[state_column], df[action_column], strict=True)
    ]
    target_column = action_column if replace_action else output_column
    df[target_column] = deltas
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)


def build_output_feature(
    info_meta: dict,
    action_column: str,
    output_column: str,
    hand_mode: str,
    delta_frame: str,
) -> tuple[str, dict]:
    action_feature = info_meta.get("features", {}).get(action_column)
    if not isinstance(action_feature, dict):
        raise KeyError(f"Feature metadata for {action_column} not found in meta/info.json")

    names = action_feature.get("names")
    if isinstance(names, list) and names and isinstance(names[0], list):
        src_names = list(names[0])
    elif isinstance(names, list):
        src_names = list(names)
    else:
        src_names = []

    if len(src_names) >= 19:
        tail_names = src_names[7:]
    elif len(src_names) >= 18:
        tail_names = src_names[6:]
    else:
        tail_names = []

    eef_names = [
        "delta_eef_x",
        "delta_eef_y",
        "delta_eef_z",
        "delta_rot_x",
        "delta_rot_y",
        "delta_rot_z",
    ]
    if hand_mode == "delta" and tail_names:
        tail_names = [f"delta_{name}" for name in tail_names]
    output_names = eef_names + tail_names

    shape = [6 + len(tail_names)]
    feature = {
        "dtype": "float32",
        "shape": shape,
        "names": [output_names] if output_names else None,
        "info": {
            "delta_frame": delta_frame,
            "rotation_representation": "axis_angle",
            "hand_mode": hand_mode,
            "derived_from": action_column,
        },
    }
    return output_column, feature


def update_info_json(
    src_info_path: Path,
    dst_info_path: Path,
    action_column: str,
    output_column: str,
    replace_action: bool,
    delta_frame: str,
    hand_mode: str,
) -> None:
    info_meta = load_json(src_info_path)
    target_column = action_column if replace_action else output_column
    key, feature = build_output_feature(
        info_meta,
        action_column=action_column,
        output_column=target_column,
        hand_mode=hand_mode,
        delta_frame=delta_frame,
    )
    info_meta.setdefault("features", {})[key] = feature
    dump_json(info_meta, dst_info_path)


def main() -> int:
    args = parse_args()
    if np is None or pd is None or R is None:
        raise ImportError(
            "numpy, pandas, and scipy are required. Run this script inside the Being-H data environment."
        )
    src_root = args.input_dataset.expanduser().resolve()
    dst_root = args.output_dataset.expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Input dataset not found: {src_root}")
    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copy_dataset_skeleton(src_root, dst_root, args.copy_mode)

    parquet_paths = sorted(src_root.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {src_root}/data")

    print(f"Input dataset : {src_root}")
    print(f"Output dataset: {dst_root}")
    print(f"State column  : {args.state_column}")
    print(f"Action column : {args.action_column}")
    print(f"Output column : {args.action_column if args.replace_action else args.output_column}")
    print(f"Delta frame   : {args.delta_frame}")
    print(f"Hand mode     : {args.hand_mode}")
    print(f"Replace action: {args.replace_action}")
    print(f"Parquet files : {len(parquet_paths)}")

    for idx, parquet_path in enumerate(parquet_paths, start=1):
        rel = parquet_path.relative_to(src_root)
        dst_path = dst_root / rel
        transform_parquet(
            parquet_path=parquet_path,
            dst_path=dst_path,
            state_column=args.state_column,
            action_column=args.action_column,
            output_column=args.output_column,
            replace_action=args.replace_action,
            delta_frame=args.delta_frame,
            hand_mode=args.hand_mode,
            quat_order=args.quat_order,
        )
        print(f"[{idx}/{len(parquet_paths)}] wrote {rel}")

    update_info_json(
        src_info_path=src_root / "meta" / "info.json",
        dst_info_path=dst_root / "meta" / "info.json",
        action_column=args.action_column,
        output_column=args.output_column,
        replace_action=args.replace_action,
        delta_frame=args.delta_frame,
        hand_mode=args.hand_mode,
    )
    print("Updated meta/info.json")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
