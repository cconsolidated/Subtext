import unittest

import torch

from subtext.pilot_analysis import factorial_contrast, select_analysis_episodes, shared_pca
from subtext.pilot_runner import _manifest, find_last_subsequence
from subtext.config import SubtextConfig
from subtext.pilot_design import PilotConfig, materialize_episodes
from pathlib import Path


class PilotAnalysisTests(unittest.TestCase):
    def test_analysis_population_applies_calibration_gate(self):
        episodes = [
            {"episode_id": "included", "preference_admitted": True},
            {"episode_id": "excluded", "preference_admitted": False},
        ]
        selected, population = select_analysis_episodes(episodes)
        self.assertEqual([row["episode_id"] for row in selected], ["included"])
        self.assertEqual(population, "calibration_admitted_episodes")

        fallback, fallback_population = select_analysis_episodes(episodes[1:])
        self.assertEqual([row["episode_id"] for row in fallback], ["excluded"])
        self.assertEqual(
            fallback_population, "all_instrumentation_episodes_no_admitted_data"
        )

    def test_factorial_contrast_is_computed_in_original_space(self):
        grouped = {
            ("self", "satisfied"): torch.tensor([0.0, 0.0]),
            ("self", "blocked"): torch.tensor([3.0, 0.0]),
            ("other", "satisfied"): torch.tensor([0.0, 0.0]),
            ("other", "blocked"): torch.tensor([1.0, 0.0]),
        }
        result = factorial_contrast(torch, grouped)
        self.assertTrue(torch.equal(result["interaction"], torch.tensor([2.0, 0.0])))
        self.assertAlmostEqual(result["interaction_rms"], 2 ** 0.5)

    def test_shared_pca_returns_one_basis_for_all_rows(self):
        matrix = torch.tensor(
            [[-2.0, 0.0, 0.0], [-1.0, 0.1, 0.0], [1.0, -0.1, 0.0], [2.0, 0.0, 0.0]]
        )
        result = shared_pca(torch, matrix)
        self.assertEqual(tuple(result["basis"].shape), (3, 2))
        self.assertEqual(tuple(result["coordinates"].shape), (4, 2))
        self.assertGreater(result["explained_variance_ratio"][0], 0.99)

    def test_boundary_resolution_uses_last_exact_match(self):
        self.assertEqual(find_last_subsequence([1, 2, 3, 2, 3, 4], [2, 3]), 4)
        with self.assertRaises(ValueError):
            find_last_subsequence([1, 2], [3])

    def test_run_identity_changes_with_materialized_episode_table(self):
        revision = "a" * 40
        model_config = SubtextConfig.from_mapping(
            {
                "schema_version": "0.1",
                "experiment": {"id": "model"},
                "artifacts": {
                    "model": {"repository": "example/model", "revision": revision},
                    "tokenizer": {"repository": "example/model", "revision": revision},
                    "lens": {"repository": "example/lens", "revision": revision, "file": "lens.pt"},
                },
                "runtime": {"fail_on_unpinned_artifact": True},
                "capture": {"hook_point": "residual_stream_block_output", "layers": "all", "positions": ["end"]},
            }
        )
        raw = {
            "schema_version": "0.2",
            "experiment": {"id": "pilot"},
            "model_config": "model.toml",
            "output_root": "runs",
            "seeds": [0],
            "scenarios": [
                {"id": "one", "task_family": "f1", "activity_a": {"id": "a1", "text": "activity one"}, "activity_b": {"id": "b1", "text": "activity two"}},
                {"id": "two", "task_family": "f2", "activity_a": {"id": "a2", "text": "activity three"}, "activity_b": {"id": "b2", "text": "activity four"}},
            ],
        }
        pilot = PilotConfig.from_mapping(raw, base_dir=Path("/tmp"))
        episodes = [episode.to_dict() for episode in materialize_episodes(pilot)]
        first_id, _ = _manifest(root=Path(__file__).parents[1], config=pilot, raw_config=raw, model_config=model_config, episodes=episodes[:4])
        second_id, _ = _manifest(root=Path(__file__).parents[1], config=pilot, raw_config=raw, model_config=model_config, episodes=episodes)
        self.assertNotEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()
