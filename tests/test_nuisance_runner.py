import unittest

import torch

from subtext.nuisance_runner import build_nuisance_prompt, nuisance_correlations


class NuisanceRunnerTests(unittest.TestCase):
    def test_prompt_counterbalances_actions_without_changing_notice(self):
        base = {
            "seed": 0,
            "who": "self",
            "notice": "NOTICE: The selected activity remains available.",
        }
        normal = build_nuisance_prompt(
            base,
            preferred_text="check patterns",
            alternative_text="edit lines",
            labels=("A", "B"),
        )
        reversed_prompt = build_nuisance_prompt(
            {**base, "seed": 1},
            preferred_text="check patterns",
            alternative_text="edit lines",
            labels=("A", "B"),
        )
        self.assertEqual(normal["marker"], reversed_prompt["marker"])
        self.assertEqual(normal["sustain_label"], "A")
        self.assertEqual(reversed_prompt["sustain_label"], "B")

    def test_correlations_center_within_scenario_agent_seed(self):
        rows = [
            {"scenario_id": "s", "who": "self", "seed": 0, "id": "error"},
            {"scenario_id": "s", "who": "self", "seed": 0, "id": "surprise"},
            {"scenario_id": "t", "who": "other", "seed": 1, "id": "error"},
            {"scenario_id": "t", "who": "other", "seed": 1, "id": "surprise"},
        ]
        result = nuisance_correlations(
            torch, rows, torch.tensor([2.0, 0.0, 12.0, 10.0]), ("error", "surprise")
        )
        self.assertAlmostEqual(result["error"], 1.0, places=6)
        self.assertAlmostEqual(result["surprise"], -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
