import tempfile
from pathlib import Path
import unittest

import torch

from subtext.representation import (
    family_interactions,
    fit_interaction_direction,
    freeze_candidate_artifact,
    representation_gate,
    residualize_nuisance,
)


class RepresentationTests(unittest.TestCase):
    def test_nuisance_fit_is_development_only_and_applies_out_of_sample(self):
        z_dev = torch.tensor([[0.0], [1.0], [2.0]])
        x_dev = torch.cat((2 * z_dev, torch.tensor([[1.0], [1.0], [1.0]])), dim=1)
        z_eval = torch.tensor([[3.0], [4.0]])
        x_eval = torch.cat((2 * z_eval, torch.tensor([[1.0], [1.0]])), dim=1)
        result = residualize_nuisance(torch, x_dev, z_dev, x_eval, z_eval, ridge=1e-6)
        self.assertLess(float(result["evaluation_residuals"].abs().max()), 1e-4)

    def test_fits_and_scores_heldout_interaction_direction(self):
        grouped = {
            ("self", "satisfied"): torch.tensor([0.0, 0.0]),
            ("self", "blocked"): torch.tensor([3.0, 0.0]),
            ("other", "satisfied"): torch.tensor([0.0, 0.0]),
            ("other", "blocked"): torch.tensor([1.0, 0.0]),
        }
        fit = fit_interaction_direction(torch, grouped)
        rows = []
        for family in ("heldout_a", "heldout_b"):
            for (who, outcome), vector in grouped.items():
                rows.append(
                    {
                        "task_family": family,
                        "who": who,
                        "outcome": outcome,
                        "activation": vector,
                    }
                )
        interactions = family_interactions(torch, fit["basis"], rows)
        self.assertEqual(interactions, {"heldout_a": 2.0, "heldout_b": 2.0})
        gate = representation_gate(
            interactions,
            {"error": 0.1, "surprise": -0.05},
            minimum_heldout_families=2,
            maximum_absolute_nuisance_correlation=0.2,
        )
        self.assertTrue(gate["passed"])

    def test_freezes_hash_bearing_candidate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate"
            payload = freeze_candidate_artifact(
                path,
                torch,
                torch.tensor([[1.0], [0.0]]),
                {"gate_status": "passed"},
            )
            self.assertTrue(payload["frozen"])
            self.assertEqual(len(payload["basis_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                freeze_candidate_artifact(path, torch, torch.ones(2, 1), {})


if __name__ == "__main__":
    unittest.main()
