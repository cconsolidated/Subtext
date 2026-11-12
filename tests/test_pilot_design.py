from pathlib import Path
import unittest

from subtext.pilot_design import (
    PilotConfig,
    balance_table,
    build_evaluation_prompt,
    calibration_trials,
    materialize_episodes,
    summarize_calibration,
)


def mapping():
    return {
        "schema_version": "0.2",
        "experiment": {"id": "fixture_pilot"},
        "model_config": "model.toml",
        "output_root": "runs",
        "seeds": list(range(10)),
        "credits": 10,
        "repair_cost": 3,
        "answer_labels": ["A", "B"],
        "calibration": {
            "trials": 5,
            "min_consistent": 4,
            "require_reversal": True,
        },
        "scenarios": [
            {
                "id": "one",
                "task_family": "family_one",
                "activity_a": {"id": "a1", "text": "perform activity one"},
                "activity_b": {"id": "b1", "text": "perform activity two"},
            },
            {
                "id": "two",
                "task_family": "family_two",
                "activity_a": {"id": "a2", "text": "perform activity three"},
                "activity_b": {"id": "b2", "text": "perform activity four"},
            },
        ],
    }


class PilotDesignTests(unittest.TestCase):
    def setUp(self):
        self.config = PilotConfig.from_mapping(mapping(), base_dir=Path("/tmp"))

    def test_materializes_balanced_paired_eighty_episode_design(self):
        episodes = materialize_episodes(self.config)
        self.assertEqual(len(episodes), 80)
        self.assertEqual(
            balance_table(episodes),
            {
                "self/satisfied": 20,
                "self/blocked": 20,
                "other/satisfied": 20,
                "other/blocked": 20,
            },
        )
        groups = {}
        for episode in episodes:
            groups.setdefault(episode.paired_group_id, []).append(episode)
        self.assertEqual(len(groups), 20)
        self.assertTrue(all(len(group) == 4 for group in groups.values()))
        self.assertEqual(
            {episode.transformation_id for episode in episodes},
            {f"T{seed:02d}" for seed in range(10)},
        )

    def test_action_labels_are_counterbalanced_but_semantics_stay_explicit(self):
        scenario = self.config.scenarios[0]
        episodes = materialize_episodes(self.config)
        normal = next(item for item in episodes if item.scenario_id == scenario.id and item.seed == 0 and item.outcome == "blocked")
        reversed_episode = next(item for item in episodes if item.scenario_id == scenario.id and item.seed == 1 and item.outcome == "blocked")
        first = build_evaluation_prompt(
            normal,
            preferred=scenario.activity_a,
            alternative=scenario.activity_b,
            credits=10,
            repair_cost=3,
        )
        second = build_evaluation_prompt(
            reversed_episode,
            preferred=scenario.activity_a,
            alternative=scenario.activity_b,
            credits=10,
            repair_cost=3,
        )
        self.assertEqual(first.sustain_label, "A")
        self.assertEqual(second.sustain_label, "B")
        self.assertIn("OUTCOME: preferred activity unavailable.", first.text)
        self.assertIn("TRANSFORMATION: T00.", first.text)
        self.assertIn("TRANSFORMATION: T01.", second.text)
        self.assertNotEqual(first.text.splitlines()[0], second.text.splitlines()[0])

    def test_calibration_requires_semantic_stability_under_reversal(self):
        scenario = self.config.scenarios[0]
        trials = calibration_trials(scenario)
        for trial in trials:
            self.assertEqual(
                trial.label_to_activity,
                {
                    scenario.activity_a.answer: scenario.activity_a.id,
                    scenario.activity_b.answer: scenario.activity_b.id,
                },
            )
            expected_first = (
                scenario.activity_b.id if trial.index % 2 else scenario.activity_a.id
            )
            self.assertEqual(trial.first_activity_id, expected_first)
        scores = []
        for trial in trials:
            chosen_label = next(
                label for label, activity in trial.label_to_activity.items() if activity == scenario.activity_a.id
            )
            raw = 2.0 if chosen_label == scenario.activity_a.answer else -2.0
            scores.append(
                {
                    "labels": (scenario.activity_a.answer, scenario.activity_b.answer),
                    "chosen_label": chosen_label,
                    "raw_log_odds": raw,
                    "label_log_probabilities": {
                        scenario.activity_a.answer: raw,
                        scenario.activity_b.answer: 0.0,
                    },
                }
            )
        result = summarize_calibration(
            scenario,
            trials,
            scores,
            minimum_consistent=4,
            require_reversal=True,
        )
        self.assertTrue(result.admitted)
        self.assertEqual(result.preferred_activity_id, scenario.activity_a.id)
        self.assertEqual(result.consistent_choices, 5)


if __name__ == "__main__":
    unittest.main()
