"""Deterministic sketches for compact residual-stream visualization."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_FINGERPRINT_BINS = 64
DEFAULT_CLIP_Z = 3.0


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def projection_spec(
    width: int,
    bins: int = DEFAULT_FINGERPRINT_BINS,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    """Return deterministic feature-hash buckets, signs, and normalizers.

    Each source coordinate contributes to exactly one output bin. Independent
    mixed hashes select its bin and sign; per-bin normalization prevents a bin
    with more assigned coordinates from appearing artificially stronger.
    """

    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("width must be a positive integer")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer of at least 2")
    if bins > width:
        raise ValueError("bins must not exceed the residual width")

    # Seed one coordinate into every bin, then hash the remainder. This keeps
    # the projection defined even for narrow fixture models while retaining a
    # stable mixed assignment for real residual widths.
    buckets = tuple(
        index if index < bins else _mix32(index + 0x9E3779B9) % bins
        for index in range(width)
    )
    signs = tuple(
        1 if _mix32(index + 0x85EBCA6B) & 1 else -1 for index in range(width)
    )
    counts = [0] * bins
    for bucket in buckets:
        counts[bucket] += 1
    scales = tuple(1.0 / math.sqrt(counts[bucket]) for bucket in buckets)
    return buckets, signs, scales


def build_projection(
    torch_module: Any,
    width: int,
    *,
    bins: int = DEFAULT_FINGERPRINT_BINS,
    device: str,
) -> Any:
    """Build the fixed sparse signed-hash projection as a dense device tensor."""

    buckets, signs, scales = projection_spec(width, bins)
    projection = torch_module.zeros((width, bins), dtype=torch_module.float32)
    rows = torch_module.arange(width, dtype=torch_module.long)
    columns = torch_module.tensor(buckets, dtype=torch_module.long)
    values = torch_module.tensor(
        [sign * scale for sign, scale in zip(signs, scales)],
        dtype=torch_module.float32,
    )
    projection[rows, columns] = values
    return projection.to(device)


def quantized_fingerprint(
    hidden: Any,
    projection: Any,
    *,
    clip_z: float = DEFAULT_CLIP_Z,
) -> tuple[float, list[int]]:
    """Project one residual vector and return its RMS plus signed int8 values."""

    if clip_z <= 0:
        raise ValueError("clip_z must be positive")
    vector = hidden.float()
    rms = vector.square().mean().sqrt()
    normalized = (vector @ projection) / rms.clamp_min(1e-12)
    rounded = (
        normalized.clamp(-clip_z, clip_z)
        .mul(127.0 / clip_z)
        .round()
        .cpu()
        .tolist()
    )
    quantized = [int(value) for value in rounded]
    return round(float(rms.item()), 6), quantized


def quantized_delta_fingerprint(
    source: Any,
    target: Any,
    projection: Any,
    *,
    clip_z: float = DEFAULT_CLIP_Z,
) -> tuple[float, list[int]]:
    """Project an update formed before normalization and quantization."""

    return quantized_fingerprint(
        target.float() - source.float(),
        projection,
        clip_z=clip_z,
    )
