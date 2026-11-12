import unittest

import torch

from subtext.interventions import (
    InterventionGateError,
    ResidualSubspaceIntervention,
    ensure_intervention_ready,
    match_delta_norm,
    random_subspace_like,
    relative_norm_mismatch,
    scrambled_subspace,
    source_patch,
    suppress_subspace,
)


READY = {
    "frozen": True,
    "gate_status": "passed",
    "heldout_generalization_passed": True,
    "nuisance_controls_passed": True,
    "heldout_family_count": 2,
}


class InterventionTests(unittest.TestCase):
    def test_suppression_and_source_patch_algebra(self):
        basis = torch.tensor([[1.0], [0.0]])
        vector = torch.tensor([3.0, 4.0])
        self.assertTrue(
            torch.equal(suppress_subspace(torch, vector, basis, 1.0), torch.tensor([0.0, 4.0]))
        )
        source = torch.tensor([7.0, 9.0])
        self.assertTrue(
            torch.equal(source_patch(torch, vector, source, basis, 1.0), torch.tensor([7.0, 4.0]))
        )

    def test_random_and_scrambled_shams_are_orthonormal(self):
        basis = torch.eye(5)[:, :2]
        for sham in (random_subspace_like(torch, basis, 7), scrambled_subspace(torch, basis, 7)):
            self.assertTrue(torch.allclose(sham.T @ sham, torch.eye(2), atol=1e-5))

    def test_norm_matching_and_batch_tolerance_metric(self):
        original = torch.tensor([0.0, 0.0])
        edited = torch.tensor([3.0, 4.0])
        matched = match_delta_norm(torch, original, edited, 2.0)
        self.assertAlmostEqual(float(matched.norm()), 2.0, places=6)
        self.assertAlmostEqual(relative_norm_mismatch([2.0, 2.0], [1.9, 2.1]), 0.0)

    def test_interventions_fail_closed_until_gate_passes(self):
        with self.assertRaises(InterventionGateError):
            ensure_intervention_ready({"frozen": True})
        intervention = ResidualSubspaceIntervention(
            identity="candidate-suppression",
            torch_module=torch,
            operation="suppression",
            basis=torch.tensor([[1.0], [0.0]]),
            dose=1.0,
            target_layers=(1,),
            target_position_names=("decision",),
            candidate_metadata=READY,
        )
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        untouched = intervention.edit(hidden, layer=0, positions={"decision": 1})
        edited = intervention.edit(hidden, layer=1, positions={"decision": 1})
        self.assertTrue(torch.equal(untouched, hidden))
        self.assertTrue(torch.equal(edited[0, 1], torch.tensor([0.0, 4.0])))


if __name__ == "__main__":
    unittest.main()
