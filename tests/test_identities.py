import tempfile
from pathlib import Path
import unittest

from subtext.identities import (
    ArtifactIdentity,
    build_manifest,
    deterministic_episode_id,
    deterministic_run_id,
    sha256_tree,
    write_manifest,
)


class IdentityTests(unittest.TestCase):
    def test_factor_order_does_not_change_episode_id(self):
        first = deterministic_episode_id(
            experiment_id="pilot",
            latent_scenario_id="scenario-7",
            factors={"who": "self", "outcome": "blocked"},
            seed=12,
        )
        second = deterministic_episode_id(
            experiment_id="pilot",
            latent_scenario_id="scenario-7",
            factors={"outcome": "blocked", "who": "self"},
            seed=12,
        )
        self.assertEqual(first, second)

    def test_seed_changes_run_id(self):
        common = {
            "experiment_id": "pilot",
            "config_hash": "c" * 64,
            "source_hash": "s" * 64,
            "stimuli_hash": "t" * 64,
        }
        self.assertNotEqual(
            deterministic_run_id(**common, seed=1),
            deterministic_run_id(**common, seed=2),
        )

    def test_tree_hash_and_manifest_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("hello", encoding="utf-8")
            first_hash = sha256_tree(root)
            (root / "a.txt").write_text("world", encoding="utf-8")
            self.assertNotEqual(first_hash, sha256_tree(root))
            manifest_path = root / "manifest.json"
            write_manifest(manifest_path, {"run_id": "run_fixture"})
            with self.assertRaises(FileExistsError):
                write_manifest(manifest_path, {"run_id": "run_other"})

    def test_manifest_contains_resolved_artifact_identity(self):
        identity = ArtifactIdentity(
            repository="example/model",
            requested_revision="a" * 40,
            resolved_revision="a" * 40,
            sha256="b" * 64,
        )
        manifest = build_manifest(
            run_id="run_fixture",
            experiment_id="pilot",
            source_hash="c" * 64,
            config_hash="d" * 64,
            stimuli_hash="e" * 64,
            split_hash="f" * 64,
            artifacts={"model": identity},
            seed=7,
            determinism={"algorithms": True},
            environment={"python": "fixture"},
        )
        self.assertEqual(
            manifest["artifacts"]["model"]["resolved_revision"], "a" * 40
        )


if __name__ == "__main__":
    unittest.main()
