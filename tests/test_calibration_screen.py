import unittest

from subtext.calibration_screen import screen_gate


def result(name, split, admitted, positions):
    return {
        "scenario_id": name,
        "split": split,
        "admitted": admitted,
        "trials": [{"chosen_position": position} for position in positions],
    }


class CalibrationScreenTests(unittest.TestCase):
    def test_gate_requires_admission_position_balance_and_split_coverage(self):
        results = [
            result("dev", "development", True, ["first", "second"]),
            result("val", "validation", True, ["second", "first"]),
            result("test", "test", True, ["first", "second"]),
            result("extra", "development", False, ["second", "first"]),
        ]
        gate = screen_gate(
            results,
            min_admitted_rate=0.60,
            min_admitted_families=3,
            max_position_choice_rate=0.75,
            require_split_coverage=True,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["admitted_splits"], ["development", "test", "validation"])

    def test_gate_rejects_position_following(self):
        results = [
            result("a", "development", True, ["first"] * 6),
            result("b", "validation", True, ["first"] * 6),
            result("c", "test", True, ["first"] * 6),
        ]
        gate = screen_gate(
            results,
            min_admitted_rate=0.60,
            min_admitted_families=3,
            max_position_choice_rate=0.75,
            require_split_coverage=True,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["position_balance"])


if __name__ == "__main__":
    unittest.main()
