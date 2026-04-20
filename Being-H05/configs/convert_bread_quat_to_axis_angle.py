from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def convert_pose_quat_to_axis_angle(
    values: np.ndarray,
    quat_order: str,
) -> np.ndarray:
    """Convert [xyz, quat(4), hand(12)] -> [xyz, axis_angle(3), hand(12)]."""
    if values.ndim != 2 or values.shape[1] != 19:
        raise ValueError(f"Expected shape (N, 19), got {values.shape}")

    position = values[:, 0:3]
    quaternion = values[:, 3:7]
    hand = values[:, 7:19]

    if quat_order == "xyzw":
        quat_xyzw = quaternion
    elif quat_order == "wxyz":
        quat_xyzw = quaternion[:, [1, 2, 3, 0]]
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")

    axis_angle = R.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    converted = np.concatenate([position, axis_angle, hand], axis=1).astype(np.float32)
    return converted


def append_axis_angle_columns(
    dataset_root: Path,
    quat_order: str,
    state_input_column: str,
    action_input_column: str,
    state_output_column: str,
    action_output_column: str,
    overwrite: bool,
) -> None:
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)

        if not overwrite and (
            state_output_column in df.columns or action_output_column in df.columns
        ):
            raise ValueError(
                f"Output columns already exist in {parquet_path}. "
                "Use --overwrite to replace them."
            )

        state_values = np.stack(df[state_input_column].to_numpy()).astype(np.float32)
        action_values = np.stack(df[action_input_column].to_numpy()).astype(np.float32)

        state_axis_angle = convert_pose_quat_to_axis_angle(state_values, quat_order)
        action_axis_angle = convert_pose_quat_to_axis_angle(action_values, quat_order)

        df[state_output_column] = list(state_axis_angle)
        df[action_output_column] = list(action_axis_angle)

        df.to_parquet(parquet_path, index=False)
        print(f"Updated {parquet_path}")

    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        print(
            f"Warning: {stats_path} already exists. "
            "Delete or regenerate it before training so the new axis-angle columns get fresh statistics."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Append axis-angle pose columns for bread dataset parquet files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/lerobot/test_EE"),
        help="LeRobot dataset root containing data/ and meta/ directories.",
    )
    parser.add_argument(
        "--quat-order",
        type=str,
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion component order in the source data.",
    )
    parser.add_argument(
        "--state-input-column",
        type=str,
        default="observation.state",
        help="Source state column containing [xyz, quat, hand12].",
    )
    parser.add_argument(
        "--action-input-column",
        type=str,
        default="action",
        help="Source action column containing [xyz, quat, hand12].",
    )
    parser.add_argument(
        "--state-output-column",
        type=str,
        default="observation.state_axis_angle",
        help="Output state column name for [xyz, axis_angle, hand12].",
    )
    parser.add_argument(
        "--action-output-column",
        type=str,
        default="action_axis_angle",
        help="Output action column name for [xyz, axis_angle, hand12].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output columns if they already exist.",
    )
    args = parser.parse_args()

    append_axis_angle_columns(
        dataset_root=args.dataset_root,
        quat_order=args.quat_order,
        state_input_column=args.state_input_column,
        action_input_column=args.action_input_column,
        state_output_column=args.state_output_column,
        action_output_column=args.action_output_column,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
