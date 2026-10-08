"""Turn per-residue ligandability into a ranked list of pocket centres.

A per-residue score answers "is this residue part of a pocket". The question behind it is "where
should I dock", which needs a small ranked list of points. This is the detector-free way across
that gap: take the residues the head is confident about, group them in space, and call each group
a pocket. There is no geometric cavity detector involved, so nothing outside this package has to
be installed, and a pocket no detector would have proposed can still be expressed.

A group's centre is the score-weighted mean of its residues' heavy-atom centroids. That is a shell
of residues rather than a cavity, but the shell's centroid sits a median ~2.3 A from the ligand it
surrounds, comfortably inside the usual 4 A success criterion.

Two ways to group, and they fail differently:

``linkage``    links residues whose CA atoms lie within ``link_distance`` and calls each connected
               component a pocket — single-linkage clustering, computed as a radius graph because
               that is the same thing and far cheaper. Its failure mode is chaining: lower the
               confidence threshold and the extra residues bridge neighbouring pockets into one
               component rather than revealing more pockets.
``meanshift``  climbs the score-weighted density of those same CA positions to its modes, and each
               mode is a pocket. There is no linkage to chain, so two adjacent pockets stay two
               modes, and a residue the head is confident about pulls harder than one it is unsure
               of. The bandwidth sets the scale instead of a cutoff distance.

**`meanshift` is the published clustering decoder**, at ``score_threshold=0.3``,
``bandwidth=14.0`` and ``min_residues=3`` — the setting the paper selected by grid search on
validation and froze before reading test. Those are the defaults in `zpocket eval`. `linkage`
is kept because it is cheaper and easier to reason about, but it is not what the published
localization numbers were produced with, and it scores below `meanshift` at the published
setting (DCC top-N+2 of 0.487 against 0.523 on the test split).

Bandwidth matters more than it looks: mean shift over-fragments badly when the bandwidth is
narrower than a pocket's residue shell, so small values are not a mild mis-setting but a
qualitatively different (and much worse) decoder.

The paper additionally reports `rescore` and `hybrid` decoders, which re-rank an external
detector's cavities (fpocket, P2Rank) and score higher than clustering alone. They are left out
here deliberately: they need those tools installed and run per fold, and this package is meant to
run on a head checkpoint and a fold and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from zpocket.discrete.structure import Fold

logger = logging.getLogger(__name__)

CLUSTER_METHODS = ("linkage", "meanshift")


@dataclass(frozen=True)
class Candidate:
    """One predicted pocket: where it is, and what the head thought of the residues in it."""

    center: np.ndarray  # (3,) fold frame
    positions: np.ndarray  # canonical positions of the residues that formed it
    score: float
    rank: int  # 1-based rank in the emitted list, best first
    source: str
    descriptors: dict = field(default_factory=dict)


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Sigmoid of the head's per-residue logits, so thresholds are on a bounded scale."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def centers_of(candidates: list[Candidate]) -> np.ndarray:
    """``(K, 3)`` centres in rank order — the form the centroid metrics take."""
    if not candidates:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack([c.center for c in candidates]).astype(np.float32)


def mean_shift_modes(
    points: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
    *,
    max_iterations: int = 100,
    tolerance: float = 0.05,
) -> np.ndarray:
    """Climb a score-weighted density to its modes, returning one label per point.

    Every point seeds its own climb, moving to the weighted mean of the points inside `bandwidth`
    until it stops moving; seeds that converge within half a bandwidth of each other are the same
    mode. Unlike single-linkage there is nothing to chain through: a saddle between two pockets
    does not merge them, it just sends its seeds to opposite sides.

    Weights are the head's per-residue probabilities, so a residue the head is confident about
    pulls the mode harder than one it is unsure of — which is the point of clustering the scores
    rather than the geometry.

    Args:
        points: ``(n, 3)`` positions to cluster.
        weights: ``(n,)`` non-negative weights, one per point.
        bandwidth: Kernel radius in angstroms; sets the scale of a pocket.
        max_iterations: Cap on the climb, in case a seed oscillates between two near-equal peaks.
        tolerance: Stop a seed once it moves less than this, in angstroms.

    Returns:
        An ``(n,)`` integer array of mode labels, contiguous from 0.
    """
    seeds = points.astype(np.float64).copy()
    for _ in range(max_iterations):
        offsets = seeds[:, None, :] - points[None, :, :]
        inside = np.linalg.norm(offsets, axis=2) <= bandwidth
        # A flat kernel weighted by score: the mean of the points in range, not of all points.
        mass = (inside * weights[None, :]).sum(axis=1, keepdims=True)
        moved = np.where(
            mass > 0,
            (inside * weights[None, :]) @ points / np.maximum(mass, 1e-12),
            seeds,
        )
        shift = np.linalg.norm(moved - seeds, axis=1).max()
        seeds = moved
        if shift < tolerance:
            break

    # Merge seeds that ended up in the same basin, then relabel contiguously.
    labels = np.full(len(seeds), -1, dtype=int)
    modes: list[np.ndarray] = []
    for i, seed in enumerate(seeds):
        for label, mode in enumerate(modes):
            if np.linalg.norm(seed - mode) <= bandwidth / 2:
                labels[i] = label
                break
        else:
            labels[i] = len(modes)
            modes.append(seed)
    return labels


def cluster(
    scores: np.ndarray,
    fold: Fold,
    *,
    score_threshold: float = 0.3,
    link_distance: float = 8.0,
    min_residues: int = 3,
    fallback_residues: int = 10,
    method: str = "meanshift",
    bandwidth: float = 14.0,
) -> list[Candidate]:
    """Build pockets directly from the confident residues, with no detector involved.

    Residues scoring at or above `score_threshold` are grouped by `method`, and each group becomes
    a pocket.

    Args:
        scores: ``(L,)`` per-residue probabilities, indexed by ``position - 1``.
        fold: The protein's fold, in the frame the centres come out in.
        score_threshold: Probability at or above which a residue is confident.
        link_distance: CA-CA distance at which two confident residues are linked. ``linkage`` only.
        min_residues: Groups smaller than this are dropped as noise.
        fallback_residues: When `score_threshold` selects nothing, fall back to this many
            top-scoring residues, so a protein always gets a prediction and cannot be silently
            skipped — a detector that proposes nothing must not be rewarded for it.
        method: ``"linkage"`` for single-linkage connected components, ``"meanshift"`` for
            score-weighted density modes; see the module docstring for why they differ.
        bandwidth: Mean-shift kernel radius in angstroms. ``meanshift`` only.

    Returns:
        Candidates ranked by summed residue score, best first.

    Raises:
        ValueError: If `method` is not one of `CLUSTER_METHODS`, or `scores` is not one value per
            residue of `fold`.
    """
    if method not in CLUSTER_METHODS:
        raise ValueError(f"unknown cluster method {method!r}; expected one of {CLUSTER_METHODS}")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != fold.length:
        raise ValueError(
            f"{fold.uniprot_id}: {scores.shape[0]} scores for a fold of length {fold.length}"
        )
    selected = np.flatnonzero(scores >= score_threshold)
    if len(selected) == 0:
        selected = np.argsort(-scores)[:fallback_residues]
    # A residue with no CA cannot be placed, so it cannot join a cluster.
    selected = selected[~np.isnan(fold.ca_coord[selected, 0])]
    if len(selected) == 0:
        return []

    coords = fold.ca_coord[selected]
    if method == "meanshift":
        labels = mean_shift_modes(coords, scores[selected], bandwidth)
    else:
        tree = cKDTree(coords)
        graph = tree.sparse_distance_matrix(tree, link_distance, output_type="coo_matrix")
        _, labels = connected_components(graph, directed=False)

    candidates = []
    for component in range(labels.max() + 1 if len(labels) else 0):
        members = selected[labels == component]
        if len(members) < min_residues:
            continue
        weights = scores[members]
        centroids = np.array([fold.centroid_of([position + 1]) for position in members])
        present = ~np.isnan(centroids[:, 0])
        if not present.any():
            continue
        center = np.average(centroids[present], axis=0, weights=weights[present])
        candidates.append(
            Candidate(
                center=center.astype(np.float32),
                positions=(members + 1).astype(np.int32),
                score=float(weights.sum()),
                rank=0,
                source="cluster",
                descriptors={"n_residues": len(members), "mean_score": float(weights.mean())},
            )
        )
    candidates.sort(key=lambda c: -c.score)
    return [replace(c, rank=rank) for rank, c in enumerate(candidates, 1)]
