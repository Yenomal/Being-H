#!/usr/bin/env python3
"""Offline action-chunk diagnostics for Being-H checkpoints on LeRobot data."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # Allow argparse --help to work in minimal shells.
    np = None


ACTION_KEYS = (
    "action.eef_position",
    "action.eef_rotation",
    "action.dexhand_position",
    "action.dexhand_position_extra",
)

STATE_CANDIDATES = (
    "observation.state_axis_angle",
    "world_abs_state",
    "observation.state",
)

ACTION_CANDIDATES = (
    "action_axis_angle",
    "world_delta_action",
    "world_abs_action",
    "action",
)

VIDEO_BACKEND_FALLBACKS = ("decord", "pyav", "torchvision_av", "opencv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Being-H checkpoint on LeRobot frames and diagnose only the "
            "returned action chunks for intra-chunk oscillation/backtracking."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot dataset root.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Checkpoint directory containing model.safetensors/config.json/tokenizer files.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional explicit metadata json path when the checkpoint directory does not contain task metadata.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional explicit tokenizer directory when the checkpoint directory does not contain tokenizer files.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for json/csv/npz diagnostics.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of frames to evaluate.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated inference calls per frame.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--llm-version",
        choices=["auto", "qwen3", "qwen2.5"],
        default="auto",
        help="Fallback LLM family when checkpoint config lacks layer_module/architectures.",
    )
    parser.add_argument("--data-config-name", type=str, default="bread")
    parser.add_argument("--dataset-name", type=str, default="bread_posttrain")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--prompt-template", choices=["short", "long"], default="long")
    parser.add_argument("--max-view-num", type=int, default=-1)
    parser.add_argument("--use-fixed-view", action="store_true")
    parser.add_argument("--task", type=str, default="", help="Override task instruction.")
    parser.add_argument("--state-column", type=str, default="auto")
    parser.add_argument("--action-column", type=str, default="auto")
    parser.add_argument(
        "--eef-action-mode",
        choices=["absolute", "relative_local", "relative_world"],
        default="absolute",
        help=(
            "How to interpret the first 6 action dims when computing chunk metrics. "
            "absolute: use raw output. relative_local/world: reconstruct EEF pose from the current state."
        ),
    )
    parser.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument(
        "--video-backend",
        choices=["decord", "pyav", "opencv", "torchvision_av"],
        default="decord",
    )
    parser.add_argument("--num-inference-timesteps", type=int, default=None)
    mpg_group = parser.add_mutually_exclusive_group()
    mpg_group.add_argument("--use-mpg", dest="use_mpg", action="store_true")
    mpg_group.add_argument("--disable-mpg", dest="use_mpg", action="store_false")
    parser.set_defaults(use_mpg=None)
    parser.add_argument("--save-chunks", action="store_true", help="Save raw predicted/GT chunks as npz.")
    return parser.parse_args()


def ensure_import_paths(model_path: Path) -> None:
    candidates = [
        model_path / "code",
        model_path.parent / "code",
        Path.cwd() / "Being-H05",
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    if importlib.util.find_spec("BeingH.inference.beingh_policy") is None:
        raise ModuleNotFoundError(
            "Cannot import BeingH.inference.beingh_policy. "
            "Expected checkpoint_parent/code or repo Being-H05 on disk."
        )


def import_runtime_modules():
    policy_module = importlib.import_module("BeingH.inference.beingh_policy")
    from BeingH.utils.video_utils import get_frames_by_timestamps

    return policy_module, get_frames_by_timestamps


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_parquet(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas with a parquet backend is required to read LeRobot parquet files.") from exc
    return pd.read_parquet(path)


def resolve_column(columns: list[str], requested: str, candidates: tuple[str, ...], kind: str) -> str:
    if requested != "auto":
        if requested not in columns:
            raise KeyError(f"Requested {kind} column '{requested}' not in parquet columns: {columns}")
        return requested
    for column in candidates:
        if column in columns:
            return column
    raise KeyError(f"Could not auto-detect {kind} column. Available columns: {columns}")


def as_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array


def quat_to_rotvec(quat: np.ndarray, order: str) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError as exc:
        raise ImportError("scipy is required when converting quaternion state/action to axis-angle.") from exc

    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if order == "wxyz":
        quat = quat[[1, 2, 3, 0]]
    return R.from_quat(quat).as_rotvec().astype(np.float32)


def to_pose18(vector: Any, quat_order: str) -> np.ndarray:
    arr = as_vector(vector)
    if arr.shape[0] >= 18 and arr.shape[0] != 19:
        return arr[:18].astype(np.float32, copy=False)
    if arr.shape[0] == 19:
        return np.concatenate(
            [arr[:3], quat_to_rotvec(arr[3:7], quat_order), arr[7:19]],
            axis=0,
        ).astype(np.float32, copy=False)
    raise ValueError(f"Expected state/action dim 18 or 19, got {arr.shape}")


def stack_pose18(values: list[Any], quat_order: str) -> np.ndarray:
    return np.stack([to_pose18(value, quat_order) for value in values], axis=0).astype(np.float32)


def get_episode_index(parquet_path: Path) -> int:
    return int(parquet_path.stem.split("_")[-1])


def video_path_for(info: dict[str, Any], dataset_root: Path, parquet_path: Path, video_key: str) -> Path:
    episode_index = get_episode_index(parquet_path)
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    template = info["video_path"]
    return dataset_root / template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def task_text_for(df, frame_idx: int, task_map: dict[int, str], override: str) -> str:
    if override:
        return override
    if "task_index" in df.columns:
        task_idx = int(np.asarray(df["task_index"].iloc[frame_idx]).reshape(-1)[0])
        return task_map.get(task_idx, f"task_index={task_idx}")
    return next(iter(task_map.values()), "Default Task")


def build_policy(args: argparse.Namespace):
    policy_module, _ = import_runtime_modules()
    BeingHPolicy = make_low_mem_policy_class(policy_module)
    patch_llm_version_detection(BeingHPolicy, args.llm_version)
    if args.prompt_template == "short":
        instruction_template = "{task_description}"
    else:
        instruction_template = (
            "According to the instruction '{task_description}', what's the micro-step actions "
            "in the next {k} steps?"
        )
        return BeingHPolicy(
        model_path=str(args.model_path.resolve()),
        data_config_name=args.data_config_name,
        dataset_name=args.dataset_name,
        embodiment_tag=args.embodiment_tag,
        instruction_template=instruction_template,
        prop_pos="front",
        max_view_num=args.max_view_num,
        use_fixed_view=args.use_fixed_view,
        device=args.device,
        use_mpg=args.use_mpg,
        num_inference_timesteps=args.num_inference_timesteps,
        enable_rtc=False,
        metadata_path=None if args.metadata_path is None else str(args.metadata_path.resolve()),
        tokenizer_path=None if args.tokenizer_path is None else str(args.tokenizer_path.resolve()),
    )


def make_low_mem_policy_class(policy_module):
    import torch

    class LowMemBeingHPolicy(policy_module.BeingHPolicy):
        VERSION_CONFIGS_FALLBACK = policy_module.VERSION_CONFIGS

        def __init__(
            self,
            model_path: str,
            data_config_name: str,
            dataset_name: str,
            embodiment_tag: str,
            instruction_template: str,
            prop_pos: str = "front",
            max_view_num: int = -1,
            use_fixed_view: bool = False,
            action_attn_mode: str = "causal",
            device: str | int = "cuda" if torch.cuda.is_available() else "cpu",
            use_mpg: bool = None,
            mpg_lambda: float = None,
            mpg_num_projections: int = None,
            mpg_refinement_iters: int = None,
            mpg_gate_temperature: float = None,
            num_inference_timesteps: int = None,
            enable_rtc: bool = True,
            metadata_variant: str = None,
            stats_selection_mode: str = "auto",
            metadata_path: str | None = None,
            tokenizer_path: str | None = None,
        ):
            self.device = torch.device(device)
            self.model_path = model_path
            self.data_config_name = data_config_name
            self.prop_pos = prop_pos
            self.action_attn_mode = action_attn_mode
            self.use_fixed_view = use_fixed_view
            self.max_view_num = max_view_num
            self.dataset_name = dataset_name
            self.metadata_variant = metadata_variant
            self.stats_selection_mode = stats_selection_mode
            self.instruction_template = instruction_template
            self.enable_rtc = enable_rtc
            self.explicit_metadata_path = metadata_path
            self.explicit_tokenizer_path = tokenizer_path

            self.embodiment_tag = policy_module.EmbodimentTag(embodiment_tag)

            DataConfigClass = policy_module.DATA_CONFIG_MAP[self.data_config_name]
            self.data_config = DataConfigClass(
                embodiment_tag=self.embodiment_tag,
                use_fixed_view=self.use_fixed_view,
                max_view_num=self.max_view_num,
                obs_indices=[0],
                action_indices=list(range(16)),
            )

            self.unified_mapping = self.data_config.UNIFIED_MAPPING
            self._modality_transform = self.data_config.get_transforms()
            self._modality_transform.eval()

            self.language_key = self.data_config.LANGUAGE_KEYS[0]
            self.num_images = len(self.data_config.VIDEO_KEYS)

            print(f"\n=== Loading model from {self.model_path} ===")
            config = policy_module.BeingHConfig.from_pretrained(self.model_path)

            llm_version = self._detect_llm_version(config.llm_config)
            print(f"Detected LLM version: {llm_version}")

            QwenConfigClass, LanguageModelClass, _ = policy_module.VERSION_CONFIGS[llm_version]
            print(f"Using {LanguageModelClass.__name__}")

            llm_config_dict = config.llm_config.to_dict()
            llm_config = QwenConfigClass.from_dict(llm_config_dict)

            expert_config_dict = llm_config_dict.get("expert_config")
            if expert_config_dict:
                if not isinstance(expert_config_dict, dict):
                    expert_config_dict = expert_config_dict.to_dict()
                expert_config = QwenConfigClass.from_dict(expert_config_dict)
                llm_config.expert_config = expert_config

            vit_config_dict = config.vit_config.to_dict()
            vit_config = policy_module.InternVisionConfig.from_dict(vit_config_dict)
            if self.device.type != "cuda":
                vit_config.use_flash_attn = False

            config.llm_config = llm_config
            config.vit_config = vit_config

            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                language_model = LanguageModelClass(config.llm_config)
                vit_model = policy_module.InternVisionModel(config.vit_config)
                connector = policy_module.InternVLConnector(
                    llm_hidden_size=config.llm_config.hidden_size,
                    vit_hidden_size=config.vit_config.hidden_size,
                    downsample_ratio=config.downsample_ratio,
                )
                self.model = policy_module.BeingH(language_model, vit_model, connector, config)
            finally:
                torch.set_default_dtype(previous_dtype)

            print("Loading state dict (streaming)...")
            self.model.to(dtype=torch.bfloat16)
            stream_load_model_weights(self.model, Path(self.model_path), dtype=torch.bfloat16)
            self.model.to(self.device, dtype=torch.bfloat16)
            self.model.eval()

            self.action_chunk_length = self.model.config.action_chunk_length
            patch_size = self.model.config.vit_config.patch_size
            self.num_image_token = self.model.num_image_token = int(
                (self.model.config.force_image_size // patch_size) ** 2 * (0.5 ** 2)
            )

            self.gen_action_type = self.model.config.gen_action_type
            self.action_token_num = self.action_chunk_length

            self._apply_mpg_overrides(
                use_mpg, mpg_lambda, mpg_num_projections, mpg_refinement_iters, mpg_gate_temperature
            )
            self._apply_flow_matching_overrides(num_inference_timesteps)
            self._setup_rtc()
            if self.explicit_tokenizer_path:
                setup_tokenizer_from_path(self, Path(self.explicit_tokenizer_path))
            else:
                self._setup_tokenizer()

            _, self.image_transform = policy_module.build_vit_transform_base(
                is_train=False,
                force_image_size=self.model.config.force_image_size,
                pad2square=False,
                normalize_type="imagenet",
            )

            if self.explicit_metadata_path:
                load_metadata_from_file(
                    self=self,
                    metadata_path=Path(self.explicit_metadata_path),
                    dataset_name=self.dataset_name,
                )
            else:
                self._load_metadata(Path(self.model_path))
            print("✓ BeingHPolicy initialized successfully")

    return LowMemBeingHPolicy


def stream_load_model_weights(model, checkpoint_dir: Path, dtype=None) -> None:
    import torch
    from safetensors import safe_open

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found under {checkpoint_dir}")

    state_refs = model.state_dict()
    model_keys = set(state_refs.keys())
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []
    invalid_keys: list[str] = []

    with torch.no_grad():
        for file_path in safetensor_files:
            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in state_refs:
                        unexpected_keys.append(key)
                        continue
                    tensor = f.get_tensor(key)
                    ref = state_refs[key]
                    if ref.shape != tensor.shape:
                        invalid_keys.append(key)
                        continue
                    target_dtype = ref.dtype if dtype is None else dtype
                    ref.copy_(tensor.to(dtype=target_dtype))
                    loaded_keys.add(key)
                    del tensor

    missing_keys = sorted(model_keys - loaded_keys)
    if missing_keys or unexpected_keys or invalid_keys:
        pieces = []
        if missing_keys:
            pieces.append(f"missing={missing_keys[:8]}")
        if unexpected_keys:
            pieces.append(f"unexpected={unexpected_keys[:8]}")
        if invalid_keys:
            pieces.append(f"invalid={invalid_keys[:8]}")
        raise RuntimeError("Streaming weight load mismatch: " + ", ".join(pieces))


def load_metadata_from_file(*, self, metadata_path: Path, dataset_name: str) -> None:
    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Explicit metadata path not found: {metadata_path}")

    print(f"[Metadata] Loading from explicit file: {metadata_path}")
    all_metadatas = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata_dict = all_metadatas.get(dataset_name)
    if metadata_dict is None:
        raise ValueError(f"No metadata found for dataset '{dataset_name}' in {metadata_path}")

    variants_key = f"{dataset_name}_variants"
    self.stats_level = "unknown"
    self.stats_source = "default"

    if variants_key in all_metadatas:
        available_variants = list(all_metadatas[variants_key].keys())
        print(f"\n[Metadata Variants] Available: {available_variants}")

        task_variants = [
            v for v in available_variants
            if all_metadatas[variants_key][v].get("stats_level") == "task"
        ]
        embodiment_variants = [
            v for v in available_variants
            if all_metadatas[variants_key][v].get("stats_level") == "embodiment"
        ]
        if task_variants:
            print(f"  Level 1 (task): {task_variants}")
        if embodiment_variants:
            print(f"  Level 2 (embodiment): {embodiment_variants}")

        if self.metadata_variant == "merged":
            print("  ✓ Using merged statistics from all variants")
            self.stats_level = metadata_dict.get("stats_level", "merged")
            self.stats_source = "merged"
        elif self.metadata_variant and self.metadata_variant != "merged":
            if self.metadata_variant in all_metadatas[variants_key]:
                metadata_dict = all_metadatas[variants_key][self.metadata_variant]
                self.stats_level = metadata_dict.get("stats_level", "unknown")
                self.stats_source = f"variant:{self.metadata_variant}"
                print(f"  ✓ Using variant: '{self.metadata_variant}' (level: {self.stats_level})")
        if not self.metadata_variant or (
            self.metadata_variant
            and self.metadata_variant not in all_metadatas[variants_key]
            and self.metadata_variant != "merged"
        ):
            if self.stats_selection_mode == "task" and task_variants:
                first_task = task_variants[0]
                metadata_dict = all_metadatas[variants_key][first_task]
                self.stats_level = "task"
                self.stats_source = f"auto:task:{first_task}"
                print(f"  → Auto-selected (task): '{first_task}'")
            elif self.stats_selection_mode == "embodiment" and embodiment_variants:
                first_emb = embodiment_variants[0]
                metadata_dict = all_metadatas[variants_key][first_emb]
                self.stats_level = "embodiment"
                self.stats_source = f"auto:embodiment:{first_emb}"
                print(f"  → Auto-selected (embodiment): '{first_emb}'")
            elif self.stats_selection_mode == "auto" and available_variants:
                first_variant = available_variants[0]
                metadata_dict = all_metadatas[variants_key][first_variant]
                self.stats_level = metadata_dict.get("stats_level", "auto")
                self.stats_source = f"auto:{first_variant}"
                print(f"  → Auto-selected: '{first_variant}' (level: {self.stats_level})")
            else:
                self.stats_level = metadata_dict.get("stats_level", "default")
                self.stats_source = "default"
                print("  → Using default top-level metadata")
    else:
        self.stats_source = "legacy:direct"

    if "embodiment_tag" in metadata_dict:
        print(f"[Metadata] Embodiment tag: {metadata_dict['embodiment_tag']}")
    print(f"[Metadata] Stats level: {self.stats_level}, Source: {self.stats_source}")

    metadata = self.__class__.__mro__[1].__dict__.get("_load_metadata")  # keep linter quiet
    del metadata
    from BeingH.utils.schema import DatasetMetadata

    dataset_metadata = DatasetMetadata.model_validate(metadata_dict)
    self._modality_transform.set_metadata(dataset_metadata)
    self.metadata = dataset_metadata


def setup_tokenizer_from_path(self, tokenizer_path: Path) -> None:
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Explicit tokenizer path not found: {tokenizer_path}")

    _, _, TokenizerClass = self.__class__.VERSION_CONFIGS_FALLBACK[self._detect_llm_version(self.model.config.llm_config)]
    tokenizer_kwargs = dict(use_fast=False, trust_remote_code=True)
    if TokenizerClass.__name__ == "AutoTokenizer":
        tokenizer_kwargs["config"] = self.model.config.llm_config
    self.tokenizer = TokenizerClass.from_pretrained(str(tokenizer_path), **tokenizer_kwargs)

    tokens = [
        '<|im_start|>', '<|im_end|>', '<img>', '</img>',
        '<|state_start|>', '<|state_end|>'
    ]
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    (self.bos_token_id, self.eos_token_id, self.start_of_image,
     self.end_of_image, self.start_of_state, self.end_of_state) = token_ids
    newline_encoded = self.tokenizer.encode('\n')
    assert len(newline_encoded) == 1, "Newline should be single token"
    self.newline_token_id = newline_encoded[0]


def patch_llm_version_detection(beingh_policy_cls, requested: str) -> None:
    """Patch older BeingHPolicy detection that can leave `version` unbound."""

    def _as_dict(config_obj):
        if config_obj is None:
            return {}
        if isinstance(config_obj, dict):
            return config_obj
        if hasattr(config_obj, "model_dump"):
            dumped = config_obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(config_obj, "to_dict"):
            dumped = config_obj.to_dict()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(config_obj, "items"):
            try:
                return dict(config_obj.items())
            except Exception:
                pass
        if hasattr(config_obj, "__dict__"):
            return {
                key: value
                for key, value in vars(config_obj).items()
                if not key.startswith("_")
            }
        if isinstance(config_obj, (str, int, float, bool)):
            return {"value": config_obj}
        if isinstance(config_obj, (list, tuple)):
            return {"value": list(config_obj)}
        out = {}
        for key in dir(config_obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(config_obj, key)
            except Exception:
                continue
            if callable(value):
                continue
            out[key] = value
        return out

    def _flatten_config_strings(config_dict: dict[str, Any]) -> str:
        values = []
        for key in ("layer_module", "model_type", "_name_or_path", "architectures"):
            value = config_dict.get(key)
            if value is not None:
                values.append(str(value))
        expert_config = config_dict.get("expert_config")
        if expert_config is not None:
            values.append(_flatten_config_strings(_as_dict(expert_config)))
        return " ".join(values).lower()

    def robust_detect_llm_version(self, llm_config):
        if requested != "auto":
            return requested

        config_dict = _as_dict(llm_config)
        haystack = _flatten_config_strings(config_dict)
        if "qwen3" in haystack:
            return "qwen3"
        if "qwen2" in haystack:
            return "qwen2.5"

        print(
            "[warn] Could not infer LLM version from config; falling back to qwen3. "
            "Use --llm-version qwen2.5 if this checkpoint is Qwen2-based."
        )
        return "qwen3"

    beingh_policy_cls._detect_llm_version = robust_detect_llm_version


def build_observation(
    *,
    policy,
    df,
    frame_idx: int,
    parquet_path: Path,
    dataset_root: Path,
    info: dict[str, Any],
    task_map: dict[int, str],
    state_column: str,
    task_override: str,
    quat_order: str,
    video_backend: str,
    get_frames_by_timestamps,
) -> dict[str, Any]:
    pose18 = to_pose18(df[state_column].iloc[frame_idx], quat_order)
    obs = {
        "state.eef_position": pose18[None, 0:3],
        "state.eef_rotation": pose18[None, 3:6],
        "state.dexhand_position": pose18[None, 6:12],
        "state.dexhand_position_extra": pose18[None, 12:18],
        "language.instruction": [task_text_for(df, frame_idx, task_map, task_override)],
    }

    timestamp = float(np.asarray(df["timestamp"].iloc[frame_idx]).reshape(-1)[0]) if "timestamp" in df.columns else None
    if timestamp is None:
        fps = float(info.get("fps", 30))
        timestamp = frame_idx / fps

    source_columns = getattr(policy.data_config, "VIDEO_SOURCE_COLUMNS", {})
    for video_key in policy.data_config.VIDEO_KEYS:
        source_column = source_columns.get(video_key)
        if source_column is None:
            if video_key.endswith("side_view"):
                source_column = "observation.images.cam_side"
            elif video_key.endswith("wrist_view"):
                source_column = "observation.images.cam_wrist"
            else:
                raise KeyError(f"No source column mapping for {video_key}")

        video_path = video_path_for(info, dataset_root, parquet_path, source_column)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for {video_key}: {video_path}")
        frame = load_frame_with_fallback(
            video_path=video_path,
            timestamp=timestamp,
            preferred_backend=video_backend,
            get_frames_by_timestamps=get_frames_by_timestamps,
        )
        obs[video_key] = np.ascontiguousarray(frame, dtype=np.uint8)[None, ...]
    return obs


def load_frame_with_fallback(
    *,
    video_path: Path,
    timestamp: float,
    preferred_backend: str,
    get_frames_by_timestamps,
) -> np.ndarray:
    tried: list[tuple[str, str]] = []
    backends = [preferred_backend] + [b for b in VIDEO_BACKEND_FALLBACKS if b != preferred_backend]
    for backend in backends:
        try:
            if backend == "pyav":
                frames = np.asarray([load_frame_with_pyav(video_path, timestamp)], dtype=np.uint8)
            else:
                frames = get_frames_by_timestamps(
                    video_path=str(video_path),
                    timestamps=np.asarray([timestamp], dtype=np.float64),
                    video_backend=backend,
                )
            if len(frames) == 0:
                raise ValueError("empty frame batch")
            if backend != preferred_backend:
                print(f"[video] fallback {preferred_backend} -> {backend} for {video_path.name}")
            return frames[0]
        except Exception as exc:
            tried.append((backend, f"{type(exc).__name__}: {exc}"))
    tried_text = "; ".join(f"{backend} => {msg}" for backend, msg in tried)
    raise RuntimeError(f"Unable to decode {video_path} at t={timestamp:.6f}. Tried: {tried_text}")


def load_frame_with_pyav(video_path: Path, timestamp: float) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise ImportError("pyav backend requested but `av` is not installed.") from exc

    target_ts = max(0.0, float(timestamp))
    best_frame = None
    best_dt = float("inf")

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                frame_ts = 0.0
            else:
                frame_ts = float(frame.pts * stream.time_base)
            dt = abs(frame_ts - target_ts)
            if dt < best_dt:
                best_dt = dt
                best_frame = frame
            if frame_ts >= target_ts and best_frame is not None:
                break

    if best_frame is None:
        raise ValueError("pyav did not decode any video frame")
    return best_frame.to_ndarray(format="rgb24")


def policy_result_to_chunk(result: dict[str, Any]) -> np.ndarray:
    chunks = []
    for key in ACTION_KEYS:
        if key not in result:
            continue
        arr = np.asarray(result[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"{key} has unexpected shape {arr.shape}")
        chunks.append(arr)
    if not chunks:
        raise KeyError(f"No action keys found in policy result. Keys={list(result.keys())}")
    lengths = {chunk.shape[0] for chunk in chunks}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent action chunk lengths: {lengths}")
    return np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)


def reconstruct_eef_chunk(
    chunk: np.ndarray,
    ref_pose18: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "absolute":
        return np.asarray(chunk, dtype=np.float32)

    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError as exc:
        raise ImportError("scipy is required for relative EEF reconstruction.") from exc

    arr = np.asarray(chunk, dtype=np.float32).copy()
    ref_pose18 = np.asarray(ref_pose18, dtype=np.float32).reshape(18)
    ref_pos = ref_pose18[:3]
    ref_rot = R.from_rotvec(ref_pose18[3:6].astype(np.float64))

    delta_pos = arr[:, :3].astype(np.float64)
    delta_rot = R.from_rotvec(arr[:, 3:6].astype(np.float64))

    if mode == "relative_local":
        abs_pos = ref_pos[None, :] + ref_rot.apply(delta_pos)
        abs_rot = ref_rot * delta_rot
    elif mode == "relative_world":
        abs_pos = ref_pos[None, :] + delta_pos
        abs_rot = delta_rot * ref_rot
    else:
        raise ValueError(f"Unsupported eef mode: {mode}")

    arr[:, :3] = abs_pos.astype(np.float32)
    arr[:, 3:6] = abs_rot.as_rotvec().astype(np.float32)
    return arr


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segment_metrics(chunk: np.ndarray, prefix: str, eps: float) -> dict[str, float | int]:
    x = np.asarray(chunk, dtype=np.float64)
    d1 = np.diff(x, axis=0)
    d2 = np.diff(x, n=2, axis=0) if x.shape[0] >= 3 else np.zeros((0, x.shape[1]))

    step_norm = np.linalg.norm(d1, axis=1) if len(d1) else np.zeros(0)
    acc_norm = np.linalg.norm(d2, axis=1) if len(d2) else np.zeros(0)
    net = float(np.linalg.norm(x[-1] - x[0])) if len(x) >= 2 else 0.0
    total_var = float(step_norm.sum())
    tv_ratio = total_var / max(net, eps)

    sign_flip_count = 0
    if len(d1) >= 2:
        big = np.abs(d1[:-1]) > eps
        sign_flip_count = int(np.logical_and(big, d1[:-1] * d1[1:] < 0).sum())

    backtrack_count = 0
    backtrack_total = 0.0
    direction = x[-1] - x[0] if len(x) >= 2 else np.zeros(x.shape[1])
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > eps and len(d1):
        projected = d1 @ (direction / direction_norm)
        backward = projected < -eps
        backtrack_count = int(backward.sum())
        backtrack_total = float((-projected[backward]).sum())

    return {
        f"{prefix}_net": net,
        f"{prefix}_tv": total_var,
        f"{prefix}_tv_ratio": tv_ratio,
        f"{prefix}_d1_mean": float(step_norm.mean()) if len(step_norm) else 0.0,
        f"{prefix}_d1_max": float(step_norm.max()) if len(step_norm) else 0.0,
        f"{prefix}_d2_mean": float(acc_norm.mean()) if len(acc_norm) else 0.0,
        f"{prefix}_d2_max": float(acc_norm.max()) if len(acc_norm) else 0.0,
        f"{prefix}_sign_flips": sign_flip_count,
        f"{prefix}_backtrack_count": backtrack_count,
        f"{prefix}_backtrack_total": backtrack_total,
    }


def chunk_metrics(chunk: np.ndarray, eps: float = 1e-6) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    metrics.update(segment_metrics(chunk[:, 0:3], "pos", eps))
    metrics.update(segment_metrics(chunk[:, 3:6], "rot", eps))
    if chunk.shape[1] > 6:
        metrics.update(segment_metrics(chunk[:, 6:], "hand", eps))
    return metrics


def repeat_metrics(chunks: list[np.ndarray]) -> dict[str, float]:
    if len(chunks) < 2:
        return {
            "repeat_pos_std_mean": 0.0,
            "repeat_pos_std_max": 0.0,
            "repeat_rot_std_mean": 0.0,
            "repeat_rot_std_max": 0.0,
        }
    stacked = np.stack(chunks, axis=0).astype(np.float64)
    std = stacked.std(axis=0)
    pos_std = np.linalg.norm(std[:, 0:3], axis=1)
    rot_std = np.linalg.norm(std[:, 3:6], axis=1)
    return {
        "repeat_pos_std_mean": float(pos_std.mean()),
        "repeat_pos_std_max": float(pos_std.max()),
        "repeat_rot_std_mean": float(rot_std.mean()),
        "repeat_rot_std_max": float(rot_std.max()),
    }


def make_gt_chunk(df, frame_idx: int, action_column: str, chunk_len: int, quat_order: str) -> np.ndarray | None:
    if action_column not in df.columns:
        return None
    rows = []
    last_idx = len(df) - 1
    for offset in range(chunk_len):
        rows.append(df[action_column].iloc[min(frame_idx + offset, last_idx)])
    return stack_pose18(rows, quat_order)


def select_samples(parquet_paths: list[Path], num_samples: int, chunk_len: int, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    candidates: list[tuple[Path, int, int]] = []
    for path in parquet_paths:
        df = read_parquet(path)
        usable_len = max(1, len(df) - chunk_len)
        if usable_len <= 1:
            frame_ids = [0]
        else:
            per_file = max(1, math.ceil(num_samples / max(1, len(parquet_paths))))
            frame_ids = np.linspace(0, usable_len - 1, per_file, dtype=int).tolist()
        for frame_idx in frame_ids:
            candidates.append((path, int(frame_idx), len(df)))
    rng.shuffle(candidates)
    return [(path, frame_idx) for path, frame_idx, _ in candidates[:num_samples]]


def risk_label(pred: dict[str, Any], gt: dict[str, Any] | None, rep: dict[str, float]) -> str:
    flags = []
    if pred["pos_backtrack_count"] > 0 or pred["rot_backtrack_count"] > 0:
        flags.append("backtrack")
    if pred["pos_tv_ratio"] > 2.5 or pred["rot_tv_ratio"] > 2.5:
        flags.append("high_tv_ratio")
    if gt is not None:
        if pred["pos_d2_max"] > max(1e-6, 2.0 * float(gt["pos_d2_max"])):
            flags.append("pos_d2_gt_x2")
        if pred["rot_d2_max"] > max(1e-6, 2.0 * float(gt["rot_d2_max"])):
            flags.append("rot_d2_gt_x2")
    if rep["repeat_pos_std_max"] > 0.002 or rep["repeat_rot_std_max"] > 0.01:
        flags.append("repeat_variance")
    return ",".join(flags) if flags else "low"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key not in {"sample_id", "frame_idx", "episode_index"}
    ]
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_max"] = float(values.max())
    risk_counts: dict[str, int] = {}
    for row in rows:
        label = str(row["risk"])
        for item in label.split(","):
            risk_counts[item] = risk_counts.get(item, 0) + 1
    summary["risk_counts"] = risk_counts
    return summary


def main() -> int:
    args = parse_args()
    if np is None:
        raise ImportError("numpy is required. Run this script inside the Being-H inference/training environment.")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = Path.cwd() / f"action_chunk_diag_{time.strftime('%Y%m%d_%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_import_paths(args.model_path)
    _, get_frames_by_timestamps = import_runtime_modules()

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    task_map = {
        int(row["task_index"]): str(row["task"])
        for row in load_jsonl(args.dataset_root / "meta" / "tasks.jsonl")
    }
    parquet_paths = sorted(args.dataset_root.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_root}/data")

    print(f"[load] checkpoint={args.model_path}")
    policy = build_policy(args)
    chunk_len = int(policy.action_chunk_length)
    samples = select_samples(parquet_paths, args.num_samples, chunk_len, args.seed)

    print(f"[data] dataset={args.dataset_root}")
    print(f"[data] samples={len(samples)} chunk_len={chunk_len} repeats={args.repeats}")

    rows: list[dict[str, Any]] = []
    saved_chunks: dict[str, np.ndarray] = {}

    for sample_id, (parquet_path, frame_idx) in enumerate(samples):
        df = read_parquet(parquet_path)
        columns = list(df.columns)
        state_column = resolve_column(columns, args.state_column, STATE_CANDIDATES, "state")
        ref_pose18 = to_pose18(df[state_column].iloc[frame_idx], args.quat_order)
        action_column = None
        try:
            action_column = resolve_column(columns, args.action_column, ACTION_CANDIDATES, "action")
        except KeyError:
            action_column = None

        obs = build_observation(
            policy=policy,
            df=df,
            frame_idx=frame_idx,
            parquet_path=parquet_path,
            dataset_root=args.dataset_root,
            info=info,
            task_map=task_map,
            state_column=state_column,
            task_override=args.task,
            quat_order=args.quat_order,
            video_backend=args.video_backend,
            get_frames_by_timestamps=get_frames_by_timestamps,
        )

        pred_chunks = []
        for repeat_idx in range(max(1, args.repeats)):
            set_seed(args.seed + sample_id * 1009 + repeat_idx)
            result = policy.get_action(dict(obs))
            pred_chunk_raw = policy_result_to_chunk(result)
            pred_chunks.append(reconstruct_eef_chunk(pred_chunk_raw, ref_pose18, args.eef_action_mode))

        pred_chunk = pred_chunks[0]
        pred_m = chunk_metrics(pred_chunk)
        rep_m = repeat_metrics(pred_chunks)

        gt_chunk = None
        gt_m = None
        if action_column is not None:
            gt_chunk = make_gt_chunk(df, frame_idx, action_column, chunk_len, args.quat_order)
            if gt_chunk is not None:
                gt_chunk = reconstruct_eef_chunk(gt_chunk, ref_pose18, args.eef_action_mode)
            gt_m = chunk_metrics(gt_chunk)

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "episode_index": get_episode_index(parquet_path),
            "frame_idx": int(frame_idx),
            "state_column": state_column,
            "action_column": action_column or "",
        }
        row.update({f"pred_{k}": v for k, v in pred_m.items()})
        row.update(rep_m)
        if gt_m is not None:
            row.update({f"gt_{k}": v for k, v in gt_m.items()})
            row["pred_gt_pos_d2_max_ratio"] = float(pred_m["pos_d2_max"]) / max(1e-6, float(gt_m["pos_d2_max"]))
            row["pred_gt_rot_d2_max_ratio"] = float(pred_m["rot_d2_max"]) / max(1e-6, float(gt_m["rot_d2_max"]))
        row["risk"] = risk_label(pred_m, gt_m, rep_m)
        rows.append(row)

        if args.save_chunks:
            saved_chunks[f"pred_{sample_id}"] = pred_chunk
            if gt_chunk is not None:
                saved_chunks[f"gt_{sample_id}"] = gt_chunk

        print(
            "[sample {sid:02d}] ep={ep} frame={frame} risk={risk} "
            "pred_pos_tv_ratio={ptr:.2f} pred_pos_d2_max={pd2:.5f} "
            "pred_rot_d2_max={rd2:.5f} repeat_pos_std_max={rps:.5f}".format(
                sid=sample_id,
                ep=row["episode_index"],
                frame=frame_idx,
                risk=row["risk"],
                ptr=row["pred_pos_tv_ratio"],
                pd2=row["pred_pos_d2_max"],
                rd2=row["pred_rot_d2_max"],
                rps=row["repeat_pos_std_max"],
            )
        )

    summary = summarize_rows(rows)
    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary,
        "rows": rows,
    }
    json_path = args.output_dir / "diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output_dir / "diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.save_chunks and saved_chunks:
        np.savez_compressed(args.output_dir / "chunks.npz", **saved_chunks)

    print("\n[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[wrote] {json_path}")
    print(f"[wrote] {csv_path}")
    if args.save_chunks:
        print(f"[wrote] {args.output_dir / 'chunks.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
