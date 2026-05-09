import ast
from pathlib import Path
import unittest


class FlowerNormalizationConfigTest(unittest.TestCase):
    @staticmethod
    def _get_class_dict_assignment(class_name: str, target_name: str):
        source = Path("configs/data_config.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for statement in node.body:
                    if not isinstance(statement, ast.Assign):
                        continue
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id == target_name:
                            return ast.literal_eval(statement.value)
        raise AssertionError(f"{class_name}.{target_name} not found")

    def test_delta_data_config_uses_q99_for_all_state_and_action_keys(self):
        expected_state_modes = {
            "state.eef_position": "q99",
            "state.eef_rotation": "q99",
            "state.dexhand_position": "q99",
            "state.dexhand_position_extra": "q99",
        }
        expected_action_modes = {
            "action.eef_position": "q99",
            "action.eef_rotation": "q99",
            "action.dexhand_position": "q99",
            "action.dexhand_position_extra": "q99",
        }

        self.assertEqual(
            self._get_class_dict_assignment("DeltaDataConfig", "state_normalization_modes"),
            expected_state_modes,
        )
        self.assertEqual(
            self._get_class_dict_assignment("DeltaDataConfig", "action_normalization_modes"),
            expected_action_modes,
        )


if __name__ == "__main__":
    unittest.main()
