"""Centroid-distance metrics for discrete pocket prediction.

A per-residue metric asks "is this residue part of a pocket". A discrete metric asks the question
a user actually has: "where should I dock". So a prediction is a ranked list of points in the
fold's frame, and it is scored by how close those points land to real ligands.

Two distances, both standard in the pocket-finding literature:

``DCC``  distance from a predicted centre to the ground-truth ligand centroid. Literal here: the
         ligand centroid is the real one, projected into the fold's frame.
``DCA*`` distance from a predicted centre to the nearest heavy atom of the residues lining the
         site. Published DCA measures to the nearest *ligand* atom; the projections behind this
         ground truth kept the ligand's centroid but not its atoms, so the lining residues stand
         in for them. A ligand lies within 5 A of every one of those residues by construction, so
         DCA* is the looser of the two and its numbers are not comparable with published DCA.

The protocol is the usual one: rank the predictions, keep the top N where N is the protein's
ground-truth site count (or N+2), and ask which sites were found. A site is checked against every
kept prediction independently, as in the P2Rank and DeepPocket papers — one prediction sitting
between two nearby sites may satisfy both.

Those rates are recall at a fixed budget, and they say nothing about what a method emits outside
it. A decoder rescoring thirty-odd cavities and one emitting a single pocket can post the same
top-N recall while being very different to use. `average_precision` is the counterweight: it reads
the whole ranked list the way object detection does, so a prediction that matches nothing is a
false positive that pushes down every true positive ranked below it. It needs one-to-one matching
— a prediction may claim only one site, and a site only one prediction — which is a *different*
matching rule from the independent one above, and deliberately so: each metric uses the rule its
definition requires. Both are reported, and the prediction counts beside them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from zpocket.discrete.sites import Site
from zpocket.discrete.structure import Fold

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 4.0  # angstroms; the standard DCC/DCA success criterion
DEFAULT_EXTRA = (0, 2)  # top-N and top-(N+2)


@dataclass(frozen=True)
class SiteTarget:
    """A ground-truth site with the atoms DCA* measures against, resolved once per site."""

    site: Site
    lining_atoms: np.ndarray  # (n, 3) heavy atoms of the lining residues, in the fold frame


def build_targets(
    sites: dict[str, list[Site]], folds: dict[str, Fold]
) -> dict[str, list[SiteTarget]]:
    """Attach each site's lining-residue heavy atoms, dropping sites whose fold is missing.

    Args:
        sites: Ground-truth sites keyed by UniProt.
        folds: Loaded folds keyed by UniProt.

    Returns:
        ``{uniprot_id: [SiteTarget, ...]}`` for proteins whose fold is available.
    """
    targets: dict[str, list[SiteTarget]] = {}
    for uid, protein_sites in sites.items():
        fold = folds.get(uid)
        if fold is None:
            continue
        resolved = [
            SiteTarget(site=site, lining_atoms=fold.atoms_of(site.positions))
            for site in protein_sites
        ]
        kept = [t for t in resolved if len(t.lining_atoms)]
        if len(kept) < len(resolved):
            logger.warning(
                "%s: %d sites have no lining atoms in the fold", uid, len(resolved) - len(kept)
            )
        if kept:
            targets[uid] = kept
    return targets


def distances_to_site(centers: np.ndarray, target: SiteTarget) -> tuple[np.ndarray, np.ndarray]:
    """DCC and DCA* from each predicted centre to one site.

    Args:
        centers: ``(K, 3)`` predicted centres, in rank order.
        target: The ground-truth site to measure against.

    Returns:
        ``(dcc, dca)``, each ``(K,)`` — distance to the ligand centroid, and to the nearest
        lining heavy atom.
    """
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(centers) == 0:
        return np.empty(0), np.empty(0)
    dcc = np.linalg.norm(centers - target.site.centroid[None, :], axis=1)
    dca = np.linalg.norm(centers[:, None, :] - target.lining_atoms[None, :, :], axis=2).min(axis=1)
    return dcc, dca


def greedy_matches(centers: np.ndarray, targets: list[SiteTarget], threshold: float) -> np.ndarray:
    """Which predictions claim a site, walking the ranked list one-to-one.

    A prediction claims the nearest site inside `threshold` that no better-ranked prediction has
    already taken; a site can be claimed once. This is the matching that precision requires — a
    single centre sitting between two sites is one true positive, not two — and it is deliberately
    stricter than the independent rule the published top-N rates use.

    Args:
        centers: ``(K, 3)`` predicted centres in rank order, best first.
        targets: The protein's ground-truth sites.
        threshold: Success distance in angstroms.

    Returns:
        A ``(K,)`` boolean array, True where that prediction claimed a site.
    """
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(centers) == 0 or not targets:
        return np.zeros(len(centers), dtype=bool)

    site_centroids = np.stack([t.site.centroid for t in targets])
    distances = np.linalg.norm(centers[:, None, :] - site_centroids[None, :, :], axis=2)
    claimed = np.zeros(len(targets), dtype=bool)
    is_match = np.zeros(len(centers), dtype=bool)
    for rank, row in enumerate(distances):
        available = np.where(~claimed & (row <= threshold))[0]
        if len(available):
            claimed[available[np.argmin(row[available])]] = True
            is_match[rank] = True
    return is_match


def drop_masked_predictions(
    centers: np.ndarray,
    scored: list[SiteTarget],
    masked: list[SiteTarget],
    threshold: float,
) -> np.ndarray:
    """Remove predictions that land only on a masked site, preserving rank order.

    A masked site is a real pocket the split does not score — typically one already public before
    the split's temporal cutoff. Counting a prediction on it as a false positive would penalise a
    method for being right, which is the opposite of what masking means; counting it as a hit
    would credit a pocket the split deliberately excludes. Dropping it does neither.

    Args:
        centers: ``(K, 3)`` predicted centres in rank order.
        scored: The protein's scored sites.
        masked: The protein's masked sites.
        threshold: Success distance in angstroms.

    Returns:
        The surviving centres, in their original order.
    """
    if not len(centers) or not masked:
        return centers
    on_masked = _within(centers, masked, threshold)
    on_scored = _within(centers, scored, threshold)
    return centers[on_scored | ~on_masked]


def _within(centers: np.ndarray, sites: list[SiteTarget], threshold: float) -> np.ndarray:
    """Which centres lie within `threshold` of any of `sites`, by DCC."""
    if not sites:
        return np.zeros(len(centers), dtype=bool)
    centroids = np.stack([t.site.centroid for t in sites])
    return (np.linalg.norm(centers[:, None, :] - centroids[None, :, :], axis=2) <= threshold).any(
        axis=1
    )


def average_precision(centers: np.ndarray, targets: list[SiteTarget], threshold: float) -> float:
    """Detection-style average precision over one protein's whole ranked list.

    A prediction that claims nothing is a false positive, which lowers the precision credited to
    every true positive ranked below it — so unlike a top-N rate, this charges for over-emitting.
    Summing precision at each true positive and dividing by the site count is the all-points
    (VOC-style) AP, which reaches 1.0 only when every site is found and nothing unmatched is ranked
    above them. On a single-site protein it reduces to the reciprocal rank of the hit, so it reads
    as a ranking-quality measure rather than a coverage one.

    Args:
        centers: ``(K, 3)`` predicted centres in rank order, best first.
        targets: The protein's ground-truth sites.
        threshold: Success distance in angstroms.

    Returns:
        AP in [0, 1]. Zero when nothing was predicted; a protein with no sites returns NaN, since
        AP is undefined without a denominator.
    """
    if not targets:
        return float("nan")
    matches = greedy_matches(centers, targets, threshold)
    if not len(matches):
        return 0.0
    found = np.cumsum(matches)
    ranks = np.arange(1, len(matches) + 1)
    return float((found[matches] / ranks[matches]).sum() / len(targets))


def _budget_label(metric: str, k: int) -> str:
    """Column name for a hit at budget ``N + k``, matching the keys `evaluate` reports."""
    return f"{metric}_top_n" + (f"_plus_{k}" if k else "")


def _score_protein(
    centers: np.ndarray,
    site_targets: list[SiteTarget],
    masked_targets: list[SiteTarget],
    threshold: float,
    extra: tuple[int, ...],
) -> tuple[list[dict], dict]:
    """Everything one protein contributes to a score, per site and in aggregate.

    Factored out so that `evaluate` and `per_site_records` cannot drift: a stratified table and
    the headline number it is supposed to decompose are built from the same rows.

    Args:
        centers: ``(K, 3)`` predicted centres in rank order, best first.
        site_targets: The protein's scored sites.
        masked_targets: Its masked sites, dropped from the ranked list before anything is counted.
        threshold: Success distance in angstroms.
        extra: How many predictions beyond N to allow.

    Returns:
        ``(site_rows, protein_row)``. One site row per ground-truth site, carrying its hits at
        each budget, whether the proposals reached it at all, and its nearest prediction; one
        protein row carrying the quantities that are only defined over a whole ranked list.
    """
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    centers = drop_masked_predictions(centers, site_targets, masked_targets, threshold)
    n_sites = len(site_targets)

    per_site = [distances_to_site(centers, t) for t in site_targets]
    site_rows = []
    for target, (dcc, dca) in zip(site_targets, per_site):
        row: dict = {
            "uniprot_id": target.site.uniprot_id,
            "site_id": target.site.site_id,
            "n_sites": n_sites,
        }
        for metric, distances in (("dcc", dcc), ("dca", dca)):
            for k in extra:
                budget = n_sites + k
                row[_budget_label(metric, k)] = bool(
                    len(distances) and distances[:budget].min() <= threshold
                )
            # No budget: did the detector propose this site at all, anywhere in its list? A
            # rescoring cannot exceed this, so a gap to it is the ranker's fault and a gap
            # above it is the proposer's.
            row[f"{metric}_ceiling"] = bool(len(distances) and distances.min() <= threshold)
            row[f"nearest_{metric}"] = float(distances.min()) if len(distances) else np.inf
        site_rows.append(row)

    matches = greedy_matches(centers, site_targets, threshold)
    protein_row = {
        "n_predicted": len(centers),
        # Did the single best-ranked prediction land on any of this protein's sites?
        "top1_protein_hit": bool(
            len(centers) and min(float(dcc[0]) for dcc, _ in per_site) <= threshold
        ),
        "ap": average_precision(centers, site_targets, threshold),
        "true_positives": int(matches.sum()),
        # Precision over the whole list: the share of everything emitted that claimed a site. It
        # is rank-invariant, so it measures how clean the proposal set is rather than how well it
        # was ordered -- which is exactly the cost a decoder pays for proposing thirty to find one.
        "claimed_fraction": float(matches.mean()) if len(centers) else None,
    }
    return site_rows, protein_row


def per_site_records(
    predictions: dict[str, np.ndarray],
    targets: dict[str, list[SiteTarget]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    extra: tuple[int, ...] = DEFAULT_EXTRA,
    masked: dict[str, list[SiteTarget]] | None = None,
) -> list[dict]:
    """The rows `evaluate` averages, before it averages them.

    Every budgeted rate `evaluate` reports is the mean of one boolean column here, so grouping
    these rows by any per-site annotation -- crypticity, novelty, how many chains the pocket
    needs -- gives that same rate restricted to a stratum. The per-protein quantities (`ap`,
    `top1_protein_hit`, the precision terms) are deliberately absent: they are defined over a
    whole ranked list and do not decompose site by site.

    Args:
        predictions: ``{uniprot_id: (K, 3) centres}`` in rank order, best first.
        targets: Ground-truth sites with their lining atoms, from `build_targets`.
        threshold: Success distance in angstroms.
        extra: How many predictions beyond N to allow.
        masked: Sites neither credited nor penalised, as in `evaluate`.

    Returns:
        One record per ground-truth site: its UniProt and site id, its protein's site count, its
        hit at each budget under DCC and DCA*, whether the proposals reached it at all, and the
        distance to its nearest surviving prediction.
    """
    rows = []
    for uid, site_targets in targets.items():
        site_rows, _ = _score_protein(
            predictions.get(uid, np.empty((0, 3))),
            site_targets,
            (masked or {}).get(uid, []),
            threshold,
            extra,
        )
        rows.extend(site_rows)
    return rows


def evaluate(
    predictions: dict[str, np.ndarray],
    targets: dict[str, list[SiteTarget]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    extra: tuple[int, ...] = DEFAULT_EXTRA,
    masked: dict[str, list[SiteTarget]] | None = None,
) -> dict:
    """Success rates for one ranked predictor over one split.

    Proteins with ground-truth sites but no predictions still count, as failures — a detector that
    proposes nothing must not be rewarded for it.

    Args:
        predictions: ``{uniprot_id: (K, 3) centres}`` in rank order, best first.
        targets: Ground-truth sites with their lining atoms, from `build_targets`.
        threshold: Success distance in angstroms.
        extra: How many predictions beyond N to allow; ``0`` is top-N, ``2`` is top-(N+2).
        masked: Sites that are neither credited nor penalised — the discrete analogue of a masked
            residue. A prediction landing on one is dropped from the ranked list before scoring,
            rather than counted as a false positive, because it found a real pocket that this
            split simply does not score. A prediction that also matches a scored site is kept, so
            masking can never cost a true positive.

    Returns:
        Per-site hit rates under DCC and DCA* at each budget; ``ap``, the per-protein average
        precision over the whole ranked list, which charges for over-emitting where the budgeted
        rates do not; the ceiling those rates are chasing (the hit rate with no budget at all,
        which is what the proposals make possible and the ranking can only spend); how many
        predictions were emitted and what share of them claimed a site; the top-1 protein hit
        rate; and the median distance from a site to its nearest prediction.
    """
    site_rows: list[dict] = []
    protein_rows: list[dict] = []
    for uid, site_targets in targets.items():
        sites, protein = _score_protein(
            predictions.get(uid, np.empty((0, 3))),
            site_targets,
            (masked or {}).get(uid, []),
            threshold,
            extra,
        )
        site_rows.extend(sites)
        protein_rows.append(protein)

    n_predicted = [p["n_predicted"] for p in protein_rows]
    per_protein_ap = [p["ap"] for p in protein_rows]
    top1_protein = [p["top1_protein_hit"] for p in protein_rows]
    claimed_fraction = [
        p["claimed_fraction"] for p in protein_rows if p["claimed_fraction"] is not None
    ]
    true_positives = sum(p["true_positives"] for p in protein_rows)
    nearest_dcc = [r["nearest_dcc"] for r in site_rows]
    nearest_dca = [r["nearest_dca"] for r in site_rows]

    n_sites_total = sum(len(v) for v in targets.values())
    result = {
        "n_proteins": len(targets),
        "n_sites": n_sites_total,
        "distance_threshold": threshold,
        "ap": float(np.nanmean(per_protein_ap)) if per_protein_ap else float("nan"),
        "mean_predictions": float(np.mean(n_predicted)) if n_predicted else 0.0,
        "median_predictions": float(np.median(n_predicted)) if n_predicted else 0.0,
        "precision_all": float(np.mean(claimed_fraction)) if claimed_fraction else float("nan"),
        "recall_all": 0.0,  # filled in below, once the site total is known
        "f1": 0.0,
        "top1_protein_hit": float(np.mean(top1_protein)) if top1_protein else float("nan"),
        "median_nearest_dcc": float(np.median(nearest_dcc)) if nearest_dcc else float("nan"),
        "median_nearest_dca": float(np.median(nearest_dca)) if nearest_dca else float("nan"),
    }
    for metric in ("dcc", "dca"):
        for k in extra:
            label = _budget_label(metric, k)
            values = [r[label] for r in site_rows]
            result[label] = float(np.mean(values)) if values else float("nan")
        ceiling = [r[f"{metric}_ceiling"] for r in site_rows]
        result[f"{metric}_ceiling"] = float(np.mean(ceiling)) if ceiling else float("nan")

    # Pooled precision and recall over every prediction emitted, and their harmonic mean. This is
    # the pair a top-N rate cannot see: recall alone is bought by emitting more, precision alone by
    # emitting less, and F1 is what a sweep can select on to get a list worth reading.
    emitted = sum(n_predicted)
    precision = true_positives / emitted if emitted else 0.0
    recall = true_positives / n_sites_total if n_sites_total else 0.0
    result["precision_pooled"] = precision
    result["recall_all"] = recall
    result["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result
