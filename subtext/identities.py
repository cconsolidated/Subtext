"""Content identities, deterministic IDs, and immutable run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable, Mapping


def canonical_json(value: Any) -> bytes:
    """Encode *value* deterministically for hashing and manifests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | Path, exclude: Iterable[str] = (".git",)) -> str:
    """Hash relative paths and bytes for every file in a source tree."""

    base = Path(root).resolve()
    excluded = set(exclude)
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(base).parts)
    )
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def stable_id(namespace: str, payload: Mapping[str, Any], length: int = 20) -> str:
    if length < 8 or length > 64:
        raise ValueError("stable ID length must be between 8 and 64")
    digest = sha256_bytes(canonical_json({"namespace": namespace, "payload": payload}))
    return f"{namespace}_{digest[:length]}"


def deterministic_run_id(
    *,
    experiment_id: str,
    config_hash: str,
    source_hash: str,
    stimuli_hash: str,
    seed: int,
) -> str:
    return stable_id(
        "run",
        {
            "experiment_id": experiment_id,
            "config_hash": config_hash,
            "source_hash": source_hash,
            "stimuli_hash": stimuli_hash,
            "seed": seed,
        },
    )


def deterministic_episode_id(
    *,
    experiment_id: str,
    latent_scenario_id: str,
    factors: Mapping[str, Any],
    seed: int,
) -> str:
    return stable_id(
        "episode",
        {
            "experiment_id": experiment_id,
            "latent_scenario_id": latent_scenario_id,
            "factors": factors,
            "seed": seed,
        },
    )


@dataclass(frozen=True)
class ArtifactIdentity:
    repository: str
    requested_revision: str
    resolved_revision: str
    sha256: str | None = None
    file: str | None = None

    def __post_init__(self) -> None:
        if not self.repository or not self.requested_revision or not self.resolved_revision:
            raise ValueError("artifact identity fields must not be empty")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.sha256)
        ):
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")


def build_manifest(
    *,
    run_id: str,
    experiment_id: str,
    source_hash: str,
    config_hash: str,
    stimuli_hash: str,
    split_hash: str,
    artifacts: Mapping[str, ArtifactIdentity],
    seed: int,
    determinism: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable portion of a run manifest.

    Mutable lifecycle fields such as start/end times and counts are added by
    the runner in a later milestone; the identity-bearing core is frozen here.
    """

    return {
        "manifest_schema_version": "0.1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "source_hash": source_hash,
        "config_hash": config_hash,
        "stimuli_hash": stimuli_hash,
        "split_hash": split_hash,
        "artifacts": {name: asdict(identity) for name, identity in sorted(artifacts.items())},
        "seed": seed,
        "determinism": dict(determinism),
        "environment": dict(environment or default_environment()),
    }


def default_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Atomically create a manifest, refusing to overwrite any existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
