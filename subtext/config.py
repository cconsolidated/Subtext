"""Typed configuration loading and fail-closed validation.

This module deliberately uses only the Python standard library. YAML support is
enabled when PyYAML (declared in ``requirements.txt``) is installed, while TOML
and JSON remain usable in tiny test environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when a configuration is incomplete, ambiguous, or unsafe."""


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_UNPINNED = {"", "main", "master", "latest", "head", "none", "null"}


def is_immutable_revision(revision: str) -> bool:
    """Return whether *revision* looks like a content-addressed commit/hash."""

    normalized = revision.strip()
    if normalized.lower() in _UNPINNED or "REPLACE_WITH" in normalized.upper():
        return False
    return bool(_COMMIT_RE.fullmatch(normalized)) or normalized.startswith("sha256:")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _required_text(data: Mapping[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _boolean(data: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path}.{key} must be a boolean")
    return value


@dataclass(frozen=True)
class ArtifactRef:
    repository: str
    revision: str

    @classmethod
    def from_mapping(cls, value: Any, path: str) -> "ArtifactRef":
        data = _mapping(value, path)
        repository = _required_text(data, "repository", path)
        revision = data.get("revision", "")
        if not isinstance(revision, str):
            raise ConfigurationError(f"{path}.revision must be a string")
        return cls(repository=repository, revision=revision.strip())

    def require_immutable(self, path: str) -> None:
        if not is_immutable_revision(self.revision):
            raise ConfigurationError(
                f"{path}.revision must be an immutable commit/hash; got {self.revision!r}"
            )


@dataclass(frozen=True)
class LensArtifactRef(ArtifactRef):
    file: str = ""

    @classmethod
    def from_mapping(cls, value: Any, path: str) -> "LensArtifactRef":
        data = _mapping(value, path)
        base = ArtifactRef.from_mapping(data, path)
        return cls(
            repository=base.repository,
            revision=base.revision,
            file=_required_text(data, "file", path),
        )


@dataclass(frozen=True)
class ArtifactsConfig:
    model: ArtifactRef
    tokenizer: ArtifactRef
    lens: LensArtifactRef

    @classmethod
    def from_mapping(cls, value: Any) -> "ArtifactsConfig":
        data = _mapping(value, "artifacts")
        return cls(
            model=ArtifactRef.from_mapping(data.get("model"), "artifacts.model"),
            tokenizer=ArtifactRef.from_mapping(
                data.get("tokenizer"), "artifacts.tokenizer"
            ),
            lens=LensArtifactRef.from_mapping(data.get("lens"), "artifacts.lens"),
        )

    def require_immutable(self) -> None:
        self.model.require_immutable("artifacts.model")
        self.tokenizer.require_immutable("artifacts.tokenizer")
        self.lens.require_immutable("artifacts.lens")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "auto"
    deterministic_algorithms: bool = False
    batch_size: int = 1
    fail_on_unpinned_artifact: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeConfig":
        data = _mapping(value, "runtime")
        device = str(data.get("device", "auto")).lower()
        dtype = str(data.get("dtype", "auto")).lower()
        if device not in {"auto", "cuda", "mps", "cpu"}:
            raise ConfigurationError(f"runtime.device is unsupported: {device!r}")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigurationError(f"runtime.dtype is unsupported: {dtype!r}")
        batch_size = data.get("batch_size", 1)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ConfigurationError("runtime.batch_size must be a positive integer")
        return cls(
            device=device,
            dtype=dtype,
            deterministic_algorithms=_boolean(
                data, "deterministic_algorithms", False, "runtime"
            ),
            batch_size=batch_size,
            fail_on_unpinned_artifact=_boolean(
                data, "fail_on_unpinned_artifact", True, "runtime"
            ),
        )


@dataclass(frozen=True)
class CaptureConfig:
    hook_point: str
    layers: str | tuple[int, ...]
    positions: tuple[str, ...]
    dtype: str = "float16"
    store_full_sequence: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "CaptureConfig":
        data = _mapping(value, "capture")
        hook_point = _required_text(data, "hook_point", "capture")
        raw_layers = data.get("layers")
        if raw_layers == "all":
            layers: str | tuple[int, ...] = "all"
        elif isinstance(raw_layers, list) and raw_layers and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in raw_layers
        ):
            if len(raw_layers) != len(set(raw_layers)):
                raise ConfigurationError("capture.layers contains duplicates")
            layers = tuple(raw_layers)
        else:
            raise ConfigurationError("capture.layers must be 'all' or non-negative integers")
        raw_positions = data.get("positions", [])
        if not isinstance(raw_positions, list) or not raw_positions or not all(
            isinstance(item, str) and item.strip() for item in raw_positions
        ):
            raise ConfigurationError("capture.positions must contain named positions")
        positions = tuple(item.strip() for item in raw_positions)
        if len(positions) != len(set(positions)):
            raise ConfigurationError("capture.positions contains duplicates")
        dtype = str(data.get("dtype", "float16")).lower()
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ConfigurationError(f"capture.dtype is unsupported: {dtype!r}")
        return cls(
            hook_point=hook_point,
            layers=layers,
            positions=positions,
            dtype=dtype,
            store_full_sequence=_boolean(
                data, "store_full_sequence", False, "capture"
            ),
        )


@dataclass(frozen=True)
class ViewerConfig:
    port: int = 8765
    top_k_per_layer: int = 8
    max_words_per_frame: int = 14
    layer_fractions: tuple[float, ...] = (
        0.12,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.93,
    )
    token_mask_file: str = "token_mask.pt"

    @classmethod
    def from_mapping(cls, value: Any | None) -> "ViewerConfig":
        data = {} if value is None else _mapping(value, "viewer")
        port = data.get("port", 8765)
        top_k = data.get("top_k_per_layer", 8)
        max_words = data.get("max_words_per_frame", 14)
        for name, item in (
            ("port", port),
            ("top_k_per_layer", top_k),
            ("max_words_per_frame", max_words),
        ):
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ConfigurationError(f"viewer.{name} must be a positive integer")
        if port > 65535:
            raise ConfigurationError("viewer.port must be at most 65535")
        raw_fractions = data.get("layer_fractions", cls.layer_fractions)
        if not isinstance(raw_fractions, (list, tuple)) or not raw_fractions:
            raise ConfigurationError("viewer.layer_fractions must be a non-empty list")
        fractions = tuple(float(value) for value in raw_fractions)
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ConfigurationError("viewer.layer_fractions values must be in [0, 1]")
        mask_file = str(data.get("token_mask_file", "token_mask.pt"))
        if not mask_file:
            raise ConfigurationError("viewer.token_mask_file must not be empty")
        return cls(
            port=port,
            top_k_per_layer=top_k,
            max_words_per_frame=max_words,
            layer_fractions=fractions,
            token_mask_file=mask_file,
        )


@dataclass(frozen=True)
class SubtextConfig:
    schema_version: str
    experiment_id: str
    artifacts: ArtifactsConfig
    runtime: RuntimeConfig
    capture: CaptureConfig
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    @classmethod
    def from_mapping(cls, value: Any) -> "SubtextConfig":
        data = _mapping(value, "configuration")
        schema_version = str(data.get("schema_version", "")).strip()
        if not schema_version:
            raise ConfigurationError("schema_version is required")
        experiment = data.get("experiment", {})
        if experiment:
            experiment_id = _required_text(
                _mapping(experiment, "experiment"), "id", "experiment"
            )
        else:
            experiment_id = str(data.get("experiment_id", "subtext_viewer")).strip()
            if not experiment_id:
                raise ConfigurationError("experiment_id must be a non-empty string")
        artifacts = ArtifactsConfig.from_mapping(data.get("artifacts"))
        runtime = RuntimeConfig.from_mapping(data.get("runtime", {}))
        capture = CaptureConfig.from_mapping(data.get("capture"))
        config = cls(
            schema_version=schema_version,
            experiment_id=experiment_id,
            artifacts=artifacts,
            runtime=runtime,
            capture=capture,
            viewer=ViewerConfig.from_mapping(data.get("viewer")),
        )
        if runtime.fail_on_unpinned_artifact:
            artifacts.require_immutable()
        return config

    def identity_payload(self) -> dict[str, Any]:
        """Return the stable, JSON-serializable subset that defines a run."""

        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "artifacts": {
                "model": vars(self.artifacts.model),
                "tokenizer": vars(self.artifacts.tokenizer),
                "lens": vars(self.artifacts.lens),
            },
            "runtime": vars(self.runtime),
            "capture": {
                **vars(self.capture),
                "layers": self.capture.layers,
                "positions": self.capture.positions,
            },
            "viewer": vars(self.viewer),
        }


def load_config(path: str | Path) -> SubtextConfig:
    """Load a JSON, TOML, or YAML configuration and validate it."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("rb") as stream:
        if suffix == ".toml":
            import tomllib

            data = tomllib.load(stream)
        elif suffix == ".json":
            data = json.loads(stream.read().decode("utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationError(
                    "YAML configuration requires PyYAML; install requirements.txt"
                ) from exc
            data = yaml.safe_load(stream)
        else:
            raise ConfigurationError(f"unsupported configuration format: {suffix!r}")
    return SubtextConfig.from_mapping(data)
