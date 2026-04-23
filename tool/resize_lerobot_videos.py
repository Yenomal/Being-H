#!/usr/bin/env python3
"""
Resize selected video streams in a LeRobot video dataset and update meta/info.json.

Examples:
    python tool/resize_lerobot_videos.py \
        --input-dataset datasets/lerobot/test_EE \
        --output-dataset datasets/lerobot/test_EE_resized \
        --camera-key observation.images.cam_side \
        --width 640 \
        --height 480

    python tool/resize_lerobot_videos.py \
        --input-dataset datasets/lerobot/test_EE \
        --output-dataset datasets/lerobot/test_EE_pad640x480 \
        --camera-key cam_side \
        --camera-key cam_wrist \
        --width 640 \
        --height 480 \
        --resize-mode pad
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize selected videos inside a LeRobot dataset and write a new dataset directory."
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        required=True,
        help="Path to the source LeRobot dataset root.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        required=True,
        help="Path to the output LeRobot dataset root.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable path or command name.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable path or command name.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=["auto", "ffmpeg", "pyav"],
        default="auto",
        help="How to decode input video. auto tries ffmpeg first and falls back to pyav.",
    )
    parser.add_argument(
        "--camera-key",
        dest="camera_keys",
        action="append",
        default=[],
        help="Video feature key to resize. Can be repeated. Supports full key or short suffix such as cam_side.",
    )
    parser.add_argument("--width", type=int, required=True, help="Target video width.")
    parser.add_argument("--height", type=int, required=True, help="Target video height.")
    parser.add_argument(
        "--resize-mode",
        choices=["pad", "stretch"],
        default="pad",
        help="pad keeps aspect ratio and letterboxes, stretch resizes directly.",
    )
    parser.add_argument(
        "--video-codec",
        default="libx264",
        help="ffmpeg video codec for resized videos, for example libx264 or libsvtav1.",
    )
    parser.add_argument(
        "--input-decoder",
        default="auto",
        help="Decoder override for ffmpeg input. Default 'auto' will prefer libdav1d for AV1 when available.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Quality control for ffmpeg encoding. Smaller is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="ffmpeg preset for the selected codec.",
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
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def require_binary(name: str) -> None:
    binary_path = Path(name)
    if binary_path.is_file():
        return
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable not found in PATH: {name}")


def get_available_decoders(ffmpeg_bin: str) -> set[str]:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-decoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    decoders: set[str] = set()
    output_text = f"{result.stdout}\n{result.stderr}"
    for line in output_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Decoders:") or stripped.startswith("------"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        flags = parts[0]
        if len(flags) == 6 and flags[0] in {"V", "A", "S"}:
            decoders.add(parts[1])
    return decoders


def build_decoder_candidates(
    input_codec: str | None,
    input_decoder: str,
    available_decoders: set[str],
) -> list[str | None]:
    if input_decoder != "auto":
        if input_decoder in {"default", "none"}:
            return [None]
        if input_decoder not in available_decoders:
            available_text = ", ".join(sorted(available_decoders))
            raise ValueError(
                f"Requested decoder '{input_decoder}' is not available in current ffmpeg. "
                f"Available decoders: {available_text}"
            )
        return [input_decoder]

    if input_codec == "av1":
        preferred = ["libdav1d", "libaom-av1", "av1"]
        candidates = [decoder for decoder in preferred if decoder in available_decoders]
        if not candidates:
            return [None]
        return candidates

    return [None]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def iter_video_feature_keys(info_meta: dict) -> list[str]:
    features = info_meta.get("features", {})
    return sorted(
        key for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    )


def normalize_camera_keys(requested_keys: list[str], available_keys: list[str]) -> list[str]:
    if not requested_keys:
        return available_keys

    normalized: list[str] = []
    for raw_key in requested_keys:
        if raw_key in available_keys:
            normalized.append(raw_key)
            continue

        suffix_matches = [key for key in available_keys if key.split(".")[-1] == raw_key]
        if len(suffix_matches) == 1:
            normalized.append(suffix_matches[0])
            continue

        raise ValueError(
            f"Unknown camera key '{raw_key}'. Available keys: {', '.join(available_keys)}"
        )

    deduped: list[str] = []
    seen = set()
    for key in normalized:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def copy_or_link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, mode: str) -> None:
    for item in src.rglob("*"):
        relative = item.relative_to(src)
        target = dst / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            copy_or_link_file(item, target, mode)


def build_scale_filter(width: int, height: int, resize_mode: str) -> str:
    if resize_mode == "stretch":
        return f"scale={width}:{height}:flags=lanczos,format=yuv420p"

    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    )


def resize_pil_image(image, width: int, height: int, resize_mode: str):
    from PIL import Image

    image = image.convert("RGB")
    if resize_mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)

    src_width, src_height = image.size
    scale = min(width / src_width, height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def encode_pil_frames_with_ffmpeg(
    ffmpeg_bin: str,
    dst_video: Path,
    frames,
    width: int,
    height: int,
    fps: float | int | None,
    video_codec: str,
    crf: int,
    preset: str,
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    output_fps = fps if fps is not None else 30
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(output_fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_video),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        frame_count = 0
        for image in frames:
            process.stdin.write(image.tobytes())
            frame_count += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg encode failed for {dst_video}.\nLast ffmpeg stderr:\n{stderr[-4000:]}"
            )
        if frame_count == 0:
            raise RuntimeError(f"No frames were decoded from source video for {dst_video}")
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stderr:
            process.stderr.close()


def run_pyav_resize(
    ffmpeg_bin: str,
    src_video: Path,
    dst_video: Path,
    width: int,
    height: int,
    fps: float | int | None,
    resize_mode: str,
    video_codec: str,
    crf: int,
    preset: str,
) -> str:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV backend requested but Python package 'av' is not installed in the current environment."
        ) from exc

    container = av.open(str(src_video))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        resolved_fps = fps
        if resolved_fps is None and stream.average_rate is not None:
            resolved_fps = float(stream.average_rate)

        def frame_generator():
            for frame in container.decode(stream):
                yield resize_pil_image(frame.to_image(), width, height, resize_mode)

        encode_pil_frames_with_ffmpeg(
            ffmpeg_bin=ffmpeg_bin,
            dst_video=dst_video,
            frames=frame_generator(),
            width=width,
            height=height,
            fps=resolved_fps,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
        )
    finally:
        container.close()

    return "pyav"


def run_ffmpeg_resize(
    ffmpeg_bin: str,
    src_video: Path,
    dst_video: Path,
    width: int,
    height: int,
    fps: float | int | None,
    input_codec: str | None,
    input_decoder: str,
    available_decoders: set[str],
    resize_mode: str,
    video_codec: str,
    crf: int,
    preset: str,
) -> str | None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    vf = build_scale_filter(width, height, resize_mode)
    decoder_candidates = build_decoder_candidates(
        input_codec=input_codec,
        input_decoder=input_decoder,
        available_decoders=available_decoders,
    )

    last_error: subprocess.CalledProcessError | None = None
    last_stderr = ""
    for decoder in decoder_candidates:
        command = [
            ffmpeg_bin,
            "-y",
            "-hwaccel",
            "none",
            "-hide_banner",
        ]
        if decoder:
            command.extend(["-c:v", decoder])
        command.extend([
            "-i",
            str(src_video),
            "-an",
            "-vf",
            vf,
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])

        if fps is not None:
            command.extend(["-r", str(fps)])

        command.append(str(dst_video))
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return decoder
        except subprocess.CalledProcessError as exc:
            last_error = exc
            last_stderr = exc.stderr[-4000:] if exc.stderr else ""
            if dst_video.exists():
                dst_video.unlink()

    if last_error is not None:
        decoder_text = ", ".join("default" if item is None else item for item in decoder_candidates)
        raise RuntimeError(
            f"Failed to resize video {src_video} with decoder candidates [{decoder_text}].\n"
            f"Last ffmpeg stderr:\n{last_stderr}"
        ) from last_error
    raise RuntimeError(f"Failed to resize video: {src_video}")


def run_resize_with_backend(
    decode_backend: str,
    ffmpeg_bin: str,
    src_video: Path,
    dst_video: Path,
    width: int,
    height: int,
    fps: float | int | None,
    input_codec: str | None,
    input_decoder: str,
    available_decoders: set[str],
    resize_mode: str,
    video_codec: str,
    crf: int,
    preset: str,
) -> str:
    if decode_backend == "ffmpeg":
        result = run_ffmpeg_resize(
            ffmpeg_bin=ffmpeg_bin,
            src_video=src_video,
            dst_video=dst_video,
            width=width,
            height=height,
            fps=fps,
            input_codec=input_codec,
            input_decoder=input_decoder,
            available_decoders=available_decoders,
            resize_mode=resize_mode,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
        )
        return result or "ffmpeg-default"

    if decode_backend == "pyav":
        return run_pyav_resize(
            ffmpeg_bin=ffmpeg_bin,
            src_video=src_video,
            dst_video=dst_video,
            width=width,
            height=height,
            fps=fps,
            resize_mode=resize_mode,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
        )

    try:
        result = run_ffmpeg_resize(
            ffmpeg_bin=ffmpeg_bin,
            src_video=src_video,
            dst_video=dst_video,
            width=width,
            height=height,
            fps=fps,
            input_codec=input_codec,
            input_decoder=input_decoder,
            available_decoders=available_decoders,
            resize_mode=resize_mode,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
        )
        return result or "ffmpeg-default"
    except Exception as ffmpeg_error:
        if dst_video.exists():
            dst_video.unlink()
        try:
            return run_pyav_resize(
                ffmpeg_bin=ffmpeg_bin,
                src_video=src_video,
                dst_video=dst_video,
                width=width,
                height=height,
                fps=fps,
                resize_mode=resize_mode,
                video_codec=video_codec,
                crf=crf,
                preset=preset,
            )
        except Exception as pyav_error:
            raise RuntimeError(
                f"Both ffmpeg and pyav decode backends failed for {src_video}.\n"
                f"ffmpeg error: {ffmpeg_error}\n"
                f"pyav error: {pyav_error}"
            ) from pyav_error


def parse_ffprobe_fps(value: str) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    return float(value)


def probe_video(ffprobe_bin: str, video_path: Path) -> dict:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        stderr_text = result.stderr.strip()
        raise RuntimeError(
            f"ffprobe found no video stream in file: {video_path}\n"
            f"ffprobe stdout: {result.stdout.strip()}\n"
            f"ffprobe stderr: {stderr_text}"
        )
    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": parse_ffprobe_fps(stream.get("avg_frame_rate", "")),
    }


def update_info_meta(
    info_meta: dict,
    resized_keys: Iterable[str],
    width: int,
    height: int,
    probe_results: dict[str, dict],
) -> dict:
    resized_key_set = set(resized_keys)
    for key in resized_key_set:
        feature = info_meta["features"][key]
        feature["shape"] = [3, height, width]

        if "info" in feature:
            info_dict = feature["info"]
        elif "video_info" in feature:
            info_dict = feature["video_info"]
        else:
            info_dict = {}
            feature["info"] = info_dict

        probe = probe_results.get(key, {})
        info_dict["video.height"] = int(probe.get("height", height))
        info_dict["video.width"] = int(probe.get("width", width))
        info_dict["video.channels"] = int(info_dict.get("video.channels", 3))
        info_dict["video.pix_fmt"] = probe.get("pix_fmt", info_dict.get("video.pix_fmt", "yuv420p"))
        if probe.get("codec_name") is not None:
            info_dict["video.codec"] = probe["codec_name"]
        if probe.get("fps") is not None:
            info_dict["video.fps"] = probe["fps"]

    return info_meta


def main() -> None:
    args = parse_args()
    require_binary(args.ffmpeg_bin)
    require_binary(args.ffprobe_bin)
    available_decoders = get_available_decoders(args.ffmpeg_bin)

    input_root = args.input_dataset.resolve()
    output_root = args.output_dataset.resolve()

    if input_root == output_root:
        raise ValueError("input-dataset and output-dataset must be different paths.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive integers.")

    info_path = input_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")

    info_meta = load_json(info_path)
    available_video_keys = iter_video_feature_keys(info_meta)
    if not available_video_keys:
        raise ValueError(f"No video features found in {info_path}")

    selected_keys = normalize_camera_keys(args.camera_keys, available_video_keys)
    selected_key_set = set(selected_keys)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    for top_level_name in ("data", "meta"):
        src_dir = input_root / top_level_name
        if src_dir.exists():
            effective_mode = "copy" if top_level_name == "meta" else args.copy_mode
            copy_tree(src_dir, output_root / top_level_name, mode=effective_mode)

    for item in input_root.iterdir():
        if item.name in {"data", "meta", "videos"}:
            continue
        dst_item = output_root / item.name
        if item.is_dir():
            copy_tree(item, dst_item, mode=args.copy_mode)
        elif item.is_file():
            copy_or_link_file(item, dst_item, mode=args.copy_mode)

    input_videos_root = input_root / "videos"
    output_videos_root = output_root / "videos"
    output_videos_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(path for path in input_videos_root.rglob("*.mp4"))
    if not video_files:
        raise ValueError(f"No mp4 videos found under {input_videos_root}")

    resized_count = 0
    copied_count = 0
    first_resized_video_for_key: dict[str, Path] = {}

    for index, src_video in enumerate(video_files, start=1):
        relative = src_video.relative_to(input_videos_root)
        dst_video = output_videos_root / relative
        if len(relative.parts) < 3:
            raise ValueError(f"Unexpected video path layout: {src_video}")

        camera_key = relative.parts[1]
        if camera_key in selected_key_set:
            src_probe = probe_video(args.ffprobe_bin, src_video)
            feature_meta = info_meta["features"].get(camera_key, {})
            info_block = feature_meta.get("info", feature_meta.get("video_info", {}))
            fps = info_block.get("video.fps")
            used_decoder = run_resize_with_backend(
                decode_backend=args.decode_backend,
                ffmpeg_bin=args.ffmpeg_bin,
                src_video=src_video,
                dst_video=dst_video,
                width=args.width,
                height=args.height,
                fps=fps,
                input_codec=src_probe.get("codec_name"),
                input_decoder=args.input_decoder,
                available_decoders=available_decoders,
                resize_mode=args.resize_mode,
                video_codec=args.video_codec,
                crf=args.crf,
                preset=args.preset,
            )
            resized_count += 1
            first_resized_video_for_key.setdefault(camera_key, dst_video)
            print(f"[{index}/{len(video_files)}] resized {relative} with decoder={used_decoder}")
        else:
            copy_or_link_file(src_video, dst_video, mode=args.copy_mode)
            copied_count += 1

    probe_results = {
        key: probe_video(args.ffprobe_bin, video_path)
        for key, video_path in first_resized_video_for_key.items()
    }

    output_info_path = output_root / "meta" / "info.json"
    output_info_meta = load_json(output_info_path)
    output_info_meta = update_info_meta(
        info_meta=output_info_meta,
        resized_keys=selected_keys,
        width=args.width,
        height=args.height,
        probe_results=probe_results,
    )
    dump_json(output_info_meta, output_info_path)

    print("Finished resizing LeRobot videos.")
    print(f"Input dataset : {input_root}")
    print(f"Output dataset: {output_root}")
    print(f"Selected keys : {', '.join(selected_keys)}")
    print(f"Target size   : {args.width}x{args.height}")
    print(f"Resize mode   : {args.resize_mode}")
    print(f"Resized videos: {resized_count}")
    print(f"Copied videos : {copied_count}")


if __name__ == "__main__":
    main()
