import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_robot_client_module():
    inference_dir = REPO_ROOT / "inference" / "inference_delta"
    if str(inference_dir) not in sys.path:
        sys.path.insert(0, str(inference_dir))

    if "camera_sdk" not in sys.modules:
        camera_sdk = types.ModuleType("camera_sdk")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

        camera_sdk.ColorMode = _Dummy
        camera_sdk.RealSenseCamera = _Dummy
        camera_sdk.RealSenseCameraConfig = _Dummy
        camera_sdk.ZedCamera = _Dummy
        camera_sdk.ZedCameraConfig = _Dummy
        camera_sdk.make_cameras_from_configs = lambda *args, **kwargs: []
        sys.modules["camera_sdk"] = camera_sdk

    return _load_module("robot_client_ws_for_test", inference_dir / "robot_client_ws.py")


def _load_make_delta_module():
    return _load_module(
        "make_lerobot_delta_action_for_test",
        REPO_ROOT / "tool" / "make_lerobot_delta_action.py",
    )


class ChunkActionRunnerTests(unittest.TestCase):
    def test_temporal_aggregation_prefers_newer_chunks(self):
        robot_client_ws = _load_robot_client_module()
        runner = robot_client_ws.ChunkActionRunner(
            query_frequency=1,
            temporal_agg=True,
            temporal_agg_decay=1.0,
        )

        old_chunk = np.zeros((2, robot_client_ws.COMMAND_DIM), dtype=np.float32)
        new_chunk = np.zeros((2, robot_client_ws.COMMAND_DIM), dtype=np.float32)
        old_chunk[1, 0] = 10.0
        new_chunk[0, 0] = 20.0

        runner.update(step_idx=0, chunk=old_chunk)
        runner.update(step_idx=1, chunk=new_chunk)

        current = runner.current_action(step_idx=1)
        self.assertGreater(
            current[0],
            15.0,
            "Temporal aggregation should give higher weight to the most recent chunk.",
        )


class MakeDeltaActionTests(unittest.TestCase):
    def test_rotation_delta_uses_relative_rotation_not_direct_subtraction(self):
        make_delta_module = _load_make_delta_module()

        state = np.zeros(18, dtype=np.float32)
        action = np.zeros(18, dtype=np.float32)
        action[5] = math.pi / 2.0

        delta = make_delta_module.make_delta_action(
            state_vec=state,
            action_vec=action,
            delta_frame="world",
            hand_mode="copy",
            quat_order="xyzw",
        )

        np.testing.assert_allclose(delta[:3], np.zeros(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(
            delta[3:6],
            np.array([0.0, 0.0, math.pi / 2.0], dtype=np.float32),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
