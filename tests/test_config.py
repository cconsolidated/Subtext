import tempfile
from pathlib import Path
import unittest

from subtext.config import ConfigurationError, SubtextConfig, load_config


def valid_mapping():
    revision = "a" * 40
    return {
        "schema_version": "0.1",
        "experiment": {"id": "fixture"},
        "artifacts": {
            "model": {"repository": "example/model", "revision": revision},
            "tokenizer": {"repository": "example/model", "revision": revision},
            "lens": {
                "repository": "example/lens",
                "revision": revision,
                "file": "lens.pt",
            },
        },
        "runtime": {"fail_on_unpinned_artifact": True},
        "capture": {
            "hook_point": "residual_stream_block_output",
            "layers": [0, 2],
            "positions": ["outcome_notice_end"],
        },
    }


class ConfigTests(unittest.TestCase):
    def test_accepts_typed_pinned_config(self):
        config = SubtextConfig.from_mapping(valid_mapping())
        self.assertEqual(config.capture.layers, (0, 2))
        self.assertEqual(config.experiment_id, "fixture")

    def test_rejects_placeholder_revision(self):
        mapping = valid_mapping()
        mapping["artifacts"]["model"]["revision"] = "REPLACE_WITH_IMMUTABLE_COMMIT"
        with self.assertRaisesRegex(ConfigurationError, "immutable commit/hash"):
            SubtextConfig.from_mapping(mapping)

    def test_loads_toml_without_optional_dependencies(self):
        revision = "b" * 40
        text = f'''schema_version = "0.1"
experiment_id = "toml_fixture"
[artifacts.model]
repository = "example/model"
revision = "{revision}"
[artifacts.tokenizer]
repository = "example/model"
revision = "{revision}"
[artifacts.lens]
repository = "example/lens"
revision = "{revision}"
file = "lens.pt"
[runtime]
fail_on_unpinned_artifact = true
[capture]
hook_point = "block_output"
layers = "all"
positions = ["boundary"]
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.toml"
            path.write_text(text, encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.capture.layers, "all")

    def test_repository_default_is_pinned_and_valid(self):
        path = Path(__file__).parents[1] / "config" / "server.default.toml"
        config = load_config(path)
        self.assertTrue(config.runtime.fail_on_unpinned_artifact)
        self.assertEqual(len(config.artifacts.model.revision), 40)

    def test_rejects_string_booleans(self):
        mapping = valid_mapping()
        mapping["runtime"]["deterministic_algorithms"] = "false"
        with self.assertRaisesRegex(ConfigurationError, "must be a boolean"):
            SubtextConfig.from_mapping(mapping)


if __name__ == "__main__":
    unittest.main()
