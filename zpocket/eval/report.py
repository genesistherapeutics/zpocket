"""One evaluation pass over saved per-residue predictions: per-target, per-pocket, per-centroid.

Predictions are scored rather than recomputed, because every policy choice below is a *scoring*
choice: one inference pass supports all of them, and each is then applied to the identical
numbers. That is also what keeps the three metrics honest against each other — they read the same
score vectors and the same site policy.

The three answer different questions:

``per_target``    AUPRC within each protein, averaged over proteins. "Given this chain, did the
                  model rank its pocket residues above the rest."
``per_pocket``    AUPRC per pocket with other pockets masked out. "Did it find *each* pocket", which
                  a multi-pocket protein can hide from the per-target number.
``per_centroid``  DCC at a 4 A threshold under a top-N and top-(N+2) budget. "Where should I dock",
                  which is the question a user actually has, and the only one that needs a fold.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from zpocket.data.pocketwatch import PocketWatch, Protein
from zpocket.eval import metrics as residue_metrics
from zpocket.eval.per_pocket import score_per_pocket

logger = logging.getLogger(__name__)


def per_target_summary(
    scores: dict[str, np.ndarray], proteins: list[Protein]
) -> tuple[dict, list[str]]:
    """Per-target AUPRC and friends over the proteins that have a prediction.

    Args:
        scores: ``{uniprot_id: (L,) score vector}``.
        proteins: The split's proteins, from `zpocket.data.load_split`.

    Returns:
        ``(summary, missing)`` — the metric bundle from `zpocket.eval.metrics.summarize`, and the
        UniProt ids that had no prediction. Missing proteins are reported rather than scored as
        zero, which would silently punish an incomplete run instead of flagging it.

    Raises:
        ValueError: If a score vector's length does not match its protein.
    """
    kept, missing = [], []
    for protein in proteins:
        vector = scores.get(protein.uniprot_id)
        if vector is None:
            missing.append(protein.uniprot_id)
            continue
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.shape[0] != protein.length:
            raise ValueError(
                f"{protein.uniprot_id}: {vector.shape[0]} scores for {protein.length} residues"
            )
        kept.append((vector, protein))
    summary = residue_metrics.summarize(
        [v for v, _ in kept],
        [p.label for _, p in kept],
        [p.mask for _, p in kept],
    )
    summary["n_proteins_with_prediction"] = len(kept)
    summary["n_proteins_missing"] = len(missing)
    return summary, missing


def evaluate_residue_level(
    scores: dict[str, np.ndarray],
    dataset: PocketWatch,
    proteins: list[Protein],
    split: str,
) -> dict:
    """Both per-residue halves — per-target and per-pocket — over one set of predictions.

    Args:
        scores: ``{uniprot_id: (L,) score vector}``.
        dataset: The benchmark.
        proteins: The split's proteins, from `zpocket.data.load_split`.
        split: Which split these are.

    Returns:
        ``{"per_target": {...}, "per_pocket": {...}, "pockets": [row, ...]}``, the last being one
        record per scored pocket so any stratification can be applied afterwards.
    """
    per_target, missing = per_target_summary(scores, proteins)
    if missing:
        logger.warning(
            "%d of %d proteins have no prediction, e.g. %s",
            len(missing),
            len(proteins),
            ", ".join(missing[:3]),
        )
    per_pocket, pocket_rows = score_per_pocket(scores, dataset, proteins, split)
    logger.info(
        "per-target auprc %.4f (n=%d)  |  per-pocket auprc %.4f (%d pockets over %d targets)",
        per_target["auprc_pt"],
        per_target["n"],
        per_pocket["auprc_pp"],
        per_pocket["n_pockets"],
        per_pocket["n_proteins"],
    )
    return {
        "per_target": per_target,
        "per_pocket": per_pocket,
        "pockets": [asdict(p) for p in pocket_rows],
    }


def evaluate_centroid_level(
    scores: dict[str, np.ndarray],
    dataset: PocketWatch,
    split: str,
    fold_for,
    *,
    site_centroids=None,
    threshold: float = 4.0,
    extra: tuple[int, ...] = (0, 2),
    method: str = "meanshift",
    score_threshold: float = 0.3,
    link_distance: float = 8.0,
    min_residues: int = 3,
    bandwidth: float = 14.0,
    restrict_to=None,
) -> dict:
    """Localization metrics: decode each score vector to centres, then measure DCC.

    The decoder is the detector-free `zpocket.discrete.decode.cluster`, so this needs nothing but
    a score vector and a fold. Every knob below is a decoder hyper-parameter and belongs on
    validation — tuning them on test is how a localization number stops meaning anything.

    Args:
        scores: ``{uniprot_id: (L,) probabilities}``.
        dataset: The benchmark, read for the site policy and the shipped centroids.
        split: Which split to score.
        fold_for: ``uniprot_id -> Fold | None``. The fold's frame must be the frame
            `site_centroids` is expressed in, or every distance is meaningless.
        site_centroids: Centroid table, as a path or frame. None uses the release's own
            ``site_centroids.parquet``, which is valid against that release's ``structures/`` and
            nothing else. For folds you generated yourself, pass a table from
            `zpocket.discrete.build_site_centroids`.
        threshold: DCC success distance in angstroms; 4.0 is the literature standard.
        extra: Prediction budgets beyond N. ``(0, 2)`` reports top-N and top-(N+2).
        method: ``"linkage"`` or ``"meanshift"``; see `zpocket.discrete.decode`.
        score_threshold: Probability at or above which a residue is confident.
        link_distance: CA-CA link distance for ``linkage``.
        min_residues: Smallest group that counts as a pocket.
        bandwidth: Mean-shift kernel radius for ``meanshift``.
        restrict_to: Restrict scoring to these UniProt ids, e.g.
            ``set(all) - PAPER_EXCLUDED_TEST_PROTEINS`` to match the paper's protein set.

    Returns:
        ``{"summary": {...}, "sites": [row, ...], "n_folds_missing": int}``. The summary carries
        ``dcc_top_n`` and ``dcc_top_n_plus_2`` among the rest; the site rows are one per scored
        pocket, so any stratification can be applied to the same numbers afterwards.
    """
    from zpocket.discrete.decode import centers_of, cluster
    from zpocket.discrete.metrics import build_targets, evaluate, per_site_records
    from zpocket.discrete.sites import load_partition, restrict

    scored_sites, masked_sites = load_partition(dataset, split, site_centroids=site_centroids)
    wanted = set(scores)
    if restrict_to is not None:
        wanted &= set(restrict_to)
    scored_sites = restrict(scored_sites, wanted)
    masked_sites = restrict(masked_sites, wanted)

    folds, missing = {}, 0
    for uid in set(scored_sites) | set(masked_sites):
        fold = fold_for(uid)
        if fold is None:
            missing += 1
            continue
        folds[uid] = fold
    if missing:
        logger.warning("%d proteins with sites have no fold and cannot be localized", missing)

    predictions = {}
    for uid, fold in folds.items():
        candidates = cluster(
            scores[uid],
            fold,
            score_threshold=score_threshold,
            link_distance=link_distance,
            min_residues=min_residues,
            method=method,
            bandwidth=bandwidth,
        )
        predictions[uid] = centers_of(candidates)

    targets = build_targets(scored_sites, folds)
    masked_targets = build_targets(masked_sites, folds)
    summary = evaluate(
        predictions, targets, threshold=threshold, extra=extra, masked=masked_targets
    )
    sites = per_site_records(
        predictions, targets, threshold=threshold, extra=extra, masked=masked_targets
    )
    logger.info(
        "DCC@%.1fA  top-N %.4f  top-N+2 %.4f  ceiling %.4f  AP %.4f  "
        "%d pockets over %d targets, %.1f predictions each",
        threshold,
        summary["dcc_top_n"],
        summary["dcc_top_n_plus_2"],
        summary["dcc_ceiling"],
        summary["ap"],
        summary["n_sites"],
        summary["n_proteins"],
        summary["mean_predictions"],
    )
    return {"summary": summary, "sites": sites, "n_folds_missing": missing}
