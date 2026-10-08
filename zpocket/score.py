"""Score proteins with pocket heads: from saved trunk features, or straight out of a fold.

Two entry points, and which one you want depends on whether you need the pair rep afterwards.

`score_features_dir` reads the NPZs ``zpocket fold`` wrote. Use it when the features are already
on disk, or when you will score several head ensembles over the same features.

`score_streaming` folds and scores in one pass and never writes ``z`` at all. That is the one to
use at benchmark scale: ``z`` is quadratic in length — ~66 MB at L=360 and ~737 MB at L=1,200 in
float16 — so persisting it for a whole split costs well over a hundred gigabytes to produce score
vectors that are a few kilobytes each. The fold itself is still written, because the centroid
metrics need it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator

import numpy as np

from zpocket.heads.pairformer import ZPairformerHead
from zpocket.predict import _read_features_npz, _sigmoid, _z_logits, load_heads

logger = logging.getLogger(__name__)

FEATURES_SUFFIX = "_trunk_predictions.npz"


def features_path(features_dir: str, record_id: str) -> str:
    """Where ``zpocket fold`` puts one record's trunk-feature NPZ."""
    return os.path.join(features_dir, f"{record_id}{FEATURES_SUFFIX}")


def score_z(z: np.ndarray, heads: list[ZPairformerHead]) -> np.ndarray:
    """Per-residue pocket probabilities ``(L,)`` from an ensemble over one pair rep.

    Each head scores the shared ``z`` once and the per-head *logits* are averaged before the
    sigmoid, which is what the training-time ensembling did; averaging probabilities instead is a
    different estimator and gives different numbers.

    Args:
        z: ``(L, L, C)`` pair representation.
        heads: One or more heads, all of the same ``pair_dim``.

    Returns:
        ``(L,)`` probabilities in [0, 1].

    Raises:
        ValueError: If `heads` is empty.
    """
    if not heads:
        raise ValueError("no heads given; pass at least one checkpoint")
    per_head = [_z_logits(z, head) for head in heads]
    logits = per_head[0] if len(per_head) == 1 else np.mean(np.stack(per_head, 0), axis=0)
    return _sigmoid(logits)


def score_features_dir(
    features_dir: str,
    heads: list[ZPairformerHead],
    record_ids: Iterable[str],
    *,
    skip_missing: bool = True,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Score every record whose feature NPZ is present in `features_dir`.

    Args:
        features_dir: Directory of ``<id>_trunk_predictions.npz`` files.
        heads: The head ensemble.
        record_ids: Records to score, typically a split's UniProt ids.
        skip_missing: Record a missing NPZ and carry on. False raises instead, which is what a
            run that must be complete should do.

    Returns:
        ``({record_id: (L,) probabilities}, missing_ids)``.

    Raises:
        FileNotFoundError: If a feature NPZ is absent and `skip_missing` is False.
    """
    scores: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for i, record_id in enumerate(record_ids, 1):
        path = features_path(features_dir, record_id)
        if not os.path.exists(path):
            if not skip_missing:
                raise FileNotFoundError(f"no trunk features for {record_id!r} at {path}")
            missing.append(record_id)
            continue
        scores[record_id] = score_z(_read_features_npz(path)["z"], heads)
        if i % 100 == 0:
            logger.info("scored %d records", i)
    if missing:
        logger.warning("%d records have no trunk features", len(missing))
    return scores, missing


def score_streaming(
    fold_and_capture: Callable[[str, str], np.ndarray],
    records: Iterable[tuple[str, str]],
    heads: list[ZPairformerHead],
) -> Iterator[tuple[str, np.ndarray]]:
    """Fold each record, score it, and release ``z`` before folding the next one.

    Args:
        fold_and_capture: ``(record_id, sequence) -> z`` of shape ``(L, L, C)``. It is expected to
            write the fold itself; only ``z`` comes back, and it is dropped after scoring.
        records: ``(record_id, sequence)`` pairs.
        heads: The head ensemble.

    Yields:
        ``(record_id, (L,) probabilities)`` as each record finishes, so a caller can persist
        incrementally and a crash costs one protein rather than the whole split.
    """
    for i, (record_id, sequence) in enumerate(records, 1):
        z = fold_and_capture(record_id, sequence)
        yield record_id, score_z(z, heads)
        del z
        if i % 25 == 0:
            logger.info("folded and scored %d records", i)


def save_scores(scores: dict[str, np.ndarray], path: str) -> str:
    """Write ``{id: (L,) probabilities}`` as a compressed NPZ keyed by record id.

    Args:
        scores: The score vectors.
        path: Output ``.npz`` path; parent directories are created.

    Returns:
        The path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in scores.items()})
    logger.info("wrote %d score vectors -> %s", len(scores), path)
    return path


def load_scores(path: str) -> dict[str, np.ndarray]:
    """Read a score NPZ back into ``{id: (L,) float64}``."""
    with np.load(path) as handle:
        return {key: np.asarray(handle[key], dtype=np.float64) for key in handle.files}


__all__ = [
    "FEATURES_SUFFIX",
    "features_path",
    "load_heads",
    "load_scores",
    "save_scores",
    "score_features_dir",
    "score_streaming",
    "score_z",
]
