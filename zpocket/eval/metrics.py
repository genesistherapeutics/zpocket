"""Per-residue pocket-prediction metrics.

The headline metric is per-target AUPRC: AUPRC computed within each protein, then averaged over
proteins. Pooling every residue across proteins instead lets large proteins dominate, so
``per_target`` is the default. An optional per-residue mask restricts scoring — passing the
complement of the already-known pocket residues yields the novel-discovery AUPRC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def r_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Precision among the top-``k`` residues, where ``k`` is the number of positives.

    A rank metric read at the operating point a user actually works at — "I will look at as
    many residues as this protein has pocket residues". Its random expectation is the positive
    fraction, the same baseline as AUPRC, so the two are read on one scale.

    Args:
        scores: Per-residue scores for one target.
        labels: Matching 0/1 labels.

    Returns:
        The fraction of the top-``k`` that is positive.
    """
    k = int(labels.sum())
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return float(labels[top].sum()) / k


def hit_at_1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Whether the single highest-scoring residue is a true positive.

    The strictest useful question for a site-finder: point at one residue, is it in a pocket.
    Random expectation is again the positive fraction.

    Args:
        scores: Per-residue scores for one target.
        labels: Matching 0/1 labels.

    Returns:
        1.0 on a hit, 0.0 otherwise.
    """
    return float(labels[int(np.argmax(scores))])


@dataclass(frozen=True)
class AUPRCResult:
    """``auprc`` is the headline value; the rest is provenance for interpreting it.

    ``n_targets_scored`` counts proteins that contributed — a protein with no positives (or no
    negatives) among its scored residues has undefined AUPRC and is skipped, so this can be less
    than ``n_targets``. ``positive_fraction`` is the AUPRC a random ranker would achieve.
    """

    auprc: float
    mode: str
    n_targets: int
    n_targets_scored: int
    n_residues_scored: int
    positive_fraction: float
    per_target: list[float] | None


def _select(scores, labels, mask) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores shape {scores.shape} != labels shape {labels.shape}")
    if mask is None:
        return scores, labels
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if mask.shape != labels.shape:
        raise ValueError(f"mask shape {mask.shape} != labels shape {labels.shape}")
    return scores[mask], labels[mask]


def _both_classes(labels: np.ndarray) -> bool:
    return bool(labels.any() and not labels.all())


def pocket_auprc(scores, labels, masks=None, *, mode: str = "per_target") -> AUPRCResult:
    """AUPRC over per-residue scores, one 1-D array per protein.

    Args:
        scores: Per-residue scores, one array per protein.
        labels: Matching 0/1 labels.
        masks: Optional per-protein boolean masks; False excludes a residue from scoring
            entirely. None scores every residue.
        mode: ``"per_target"`` averages within-protein AUPRC over proteins — the headline, since
            pooling lets large proteins dominate. ``"pooled"`` computes one AUPRC over every
            scored residue.

    Returns:
        The AUPRC with the provenance needed to read it.

    Raises:
        ValueError: If `mode` is unknown, or the three sequences differ in length.
    """
    if mode not in ("per_target", "pooled"):
        raise ValueError(f"unknown mode {mode!r}; expected 'per_target' or 'pooled'")
    masks = [None] * len(scores) if masks is None else masks
    if not len(scores) == len(labels) == len(masks):
        raise ValueError("scores, labels, and masks must have the same number of targets")

    selected = [_select(s, y, m) for s, y, m in zip(scores, labels, masks)]
    n_residues = sum(len(y) for _, y in selected)
    n_positive = sum(int(y.sum()) for _, y in selected)
    positive_fraction = n_positive / n_residues if n_residues else float("nan")

    if mode == "pooled":
        flat_scores = np.concatenate([s for s, _ in selected]) if selected else np.array([])
        flat_labels = np.concatenate([y for _, y in selected]) if selected else np.array([])
        scorable = _both_classes(flat_labels)
        return AUPRCResult(
            auprc=float(average_precision_score(flat_labels, flat_scores))
            if scorable
            else float("nan"),
            mode=mode,
            n_targets=len(selected),
            n_targets_scored=int(scorable),
            n_residues_scored=n_residues,
            positive_fraction=positive_fraction,
            per_target=None,
        )

    per_target = [float(average_precision_score(y, s)) for s, y in selected if _both_classes(y)]
    return AUPRCResult(
        auprc=float(np.mean(per_target)) if per_target else float("nan"),
        mode=mode,
        n_targets=len(selected),
        n_targets_scored=len(per_target),
        n_residues_scored=n_residues,
        positive_fraction=positive_fraction,
        per_target=per_target,
    )


def pocket_auroc(scores, labels, masks=None) -> float:
    """Per-target-averaged AUROC, a prevalence-independent check alongside AUPRC.

    Args:
        scores: Per-residue scores, one array per protein.
        labels: Matching 0/1 labels.
        masks: Optional per-protein boolean masks.

    Returns:
        Mean AUROC over proteins with both classes present, or NaN if there are none.
    """
    masks = [None] * len(scores) if masks is None else masks
    values = [
        float(roc_auc_score(y, s))
        for s, y in (_select(s, y, m) for s, y, m in zip(scores, labels, masks))
        if _both_classes(y)
    ]
    return float(np.mean(values)) if values else float("nan")


def summarize(scores, labels, masks=None) -> dict:
    """The standard reporting bundle for one method.

    Args:
        scores: Per-residue scores, one array per protein.
        labels: Matching 0/1 labels.
        masks: Optional per-protein boolean masks.

    Returns:
        ``auprc_pt``, ``auprc_pool``, ``auroc_pt``, ``r_precision_pt``, ``hit1_pt``, the
        scored-protein count ``n``, and the positive fraction — which is the value a random
        ranker reaches on AUPRC, R-precision and hit@1 alike.
    """
    per_target = pocket_auprc(scores, labels, masks, mode="per_target")
    pooled = pocket_auprc(scores, labels, masks, mode="pooled")
    masks_ = [None] * len(scores) if masks is None else masks
    selected = [_select(s, y, m) for s, y, m in zip(scores, labels, masks_)]
    rank = [(r_precision(s, y), hit_at_1(s, y)) for s, y in selected if _both_classes(y)]
    return {
        "auprc_pt": per_target.auprc,
        "auprc_pool": pooled.auprc,
        "auroc_pt": pocket_auroc(scores, labels, masks),
        "r_precision_pt": float(np.mean([r for r, _ in rank])) if rank else float("nan"),
        "hit1_pt": float(np.mean([h for _, h in rank])) if rank else float("nan"),
        "n": per_target.n_targets_scored,
        "positive_fraction": per_target.positive_fraction,
    }
