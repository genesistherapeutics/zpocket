"""Per-pocket AUPRC: score one pocket at a time, with every other pocket masked out.

The per-residue metrics in `zpocket.eval.metrics` score a whole chain at once, so a protein's
pockets are pooled into one ranking problem. That flatters a model on multi-pocket proteins:
finding the one big well-studied site and missing the other three still scores well, because the
found site's residues dominate the positives.

Scoring a pocket in isolation removes that. For each pocket the positives are its own residues,
the negatives are the chain's non-pocket residues, and **every residue belonging to another pocket
is masked out** — neither positive nor negative, exactly as an unscoreable residue is handled
elsewhere. A model must then rank each site against the background on its own merit, and a protein
with four pockets contributes four independent problems instead of one easy one.

"Another pocket" means *any* other site, including one the benchmark does not score. A masked site
is still a real pocket, so its residues are not background either — putting them back as negatives
would charge a model for finding a pocket the split deliberately excludes.

Two consequences worth holding onto when reading the numbers:

- **Not comparable to the chain-level AUPRC.** Each pocket is a smaller fraction of what it is
  scored against, so the random baseline is lower and the values come out lower even for an
  unchanged model. `positive_fraction` is that baseline and is reported alongside; `lift` divides
  through by it.
- **Single-pocket proteins are nearly unchanged.** With one pocket there is nothing to mask out,
  so the difference between the two metrics is carried entirely by multi-pocket proteins. The
  summary breaks the two groups out for that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from zpocket.data.pocketwatch import PocketWatch, Protein, _check_split
from zpocket.eval.metrics import hit_at_1, r_precision
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PocketScore:
    """One pocket's AUPRC and the provenance needed to read it."""

    pocket_site_id: str
    uniprot_id: str
    auprc: float
    auroc: float
    r_precision: float  # precision among the top-k, k = this pocket's residue count
    hit_at_1: float  # 1.0 when the single top-scoring residue belongs to this pocket
    n_positive: int
    n_negative: int
    n_masked_other: int  # residues excluded because they belong to another pocket
    pockets_in_protein: int  # scored pockets, which is what this protein contributes
    positive_fraction: float  # the AUPRC a random ranker reaches on this pocket


def load_sites(dataset: PocketWatch, split: str) -> dict[str, list[tuple[str, np.ndarray, bool]]]:
    """Every site of one split, keyed by protein, tagged with whether it is scored.

    Unscored sites are returned too: they are not scored in their own right, but their residues
    are still excluded from every other pocket's background.

    Args:
        dataset: The benchmark.
        split: Which split's sites to load.

    Returns:
        ``{uniprot_id: [(pocket_site_id, zero-based positions, is_scored), ...]}``.
    """
    _check_split(split)
    scored = dataset.scored_site_ids(split)
    frame = (
        dataset.sites()
        .filter(pl.col("split") == split)
        .select("pocket_site_id", "normalized_uniprot_id", "canonical_positions")
    )
    sites: dict[str, list[tuple[str, np.ndarray, bool]]] = {}
    for row in frame.iter_rows(named=True):
        # canonical_positions are one-based UniProt positions; arrays are indexed position - 1.
        positions = np.asarray(row["canonical_positions"], dtype=np.int64) - 1
        sites.setdefault(row["normalized_uniprot_id"], []).append(
            (row["pocket_site_id"], positions, row["pocket_site_id"] in scored)
        )
    return sites


def score_one_protein(
    scores: np.ndarray,
    protein: Protein,
    sites: list[tuple[str, np.ndarray, bool]],
) -> list[PocketScore]:
    """AUPRC for each of one protein's scored pockets, other pockets masked out.

    Args:
        scores: ``(L,)`` per-residue scores.
        protein: The protein, carrying the chain-level label and mask.
        sites: ``(pocket_site_id, zero-based positions, is_scored)`` for every site on it.

    Returns:
        One `PocketScore` per scoreable pocket. A pocket with no unmasked positives, or no
        negatives left, is skipped rather than scored as zero.

    Raises:
        ValueError: If `scores` is not one value per residue of `protein`.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape[0] != protein.length:
        raise ValueError(
            f"{protein.uniprot_id}: {scores.shape[0]} scores for {protein.length} residues"
        )
    label = np.asarray(protein.label).reshape(-1).astype(bool)
    mask = np.asarray(protein.mask).reshape(-1).astype(bool)

    membership = np.zeros((len(sites), protein.length), dtype=bool)
    for i, (_, positions, _) in enumerate(sites):
        inside = positions[(positions >= 0) & (positions < protein.length)]
        membership[i, inside] = True
    n_scored = sum(1 for _, _, is_scored in sites if is_scored)

    out = []
    for i, (site_id, _, is_scored) in enumerate(sites):
        if not is_scored:
            continue
        mine = membership[i]
        # Every other pocket's residues are excluded outright — that is the point of the metric.
        # A residue shared with another pocket stays a positive for this one.
        others = membership.any(0) & ~mine if len(sites) > 1 else np.zeros_like(mine)
        positives = mask & label & mine
        negatives = mask & ~label & ~others
        if not positives.any() or not negatives.any():
            continue
        selected = positives | negatives
        y, x = positives[selected].astype(int), scores[selected]
        n_pos, n_neg = int(positives.sum()), int(negatives.sum())
        out.append(
            PocketScore(
                pocket_site_id=site_id,
                uniprot_id=protein.uniprot_id,
                auprc=float(average_precision_score(y, x)),
                auroc=float(roc_auc_score(y, x)),
                r_precision=r_precision(x, y),
                hit_at_1=hit_at_1(x, y),
                n_positive=n_pos,
                n_negative=n_neg,
                n_masked_other=int((others & mask).sum()),
                pockets_in_protein=n_scored,
                positive_fraction=n_pos / (n_pos + n_neg),
            )
        )
    return out


def summarize_pockets(pockets: list[PocketScore]) -> dict:
    """Aggregate per-pocket scores into a reporting bundle.

    Args:
        pockets: Every scored pocket.

    Returns:
        ``auprc_pp`` (mean over pockets — the headline), ``auprc_pp_protein`` (mean over proteins
        of their pockets' mean, so a many-pocket protein does not dominate), ``lift`` (mean of each
        pocket's AUPRC over its own random baseline), counts, and the mean baseline itself.
    """
    if not pockets:
        return {
            "auprc_pp": float("nan"),
            "auprc_pp_protein": float("nan"),
            "auprc_pp_multi": float("nan"),
            "auprc_pp_single": float("nan"),
            "auroc_pp": float("nan"),
            "r_precision_pp": float("nan"),
            "hit1_pp": float("nan"),
            "lift": float("nan"),
            "n_pockets": 0,
            "n_proteins": 0,
            "positive_fraction": float("nan"),
        }
    values = np.array([p.auprc for p in pockets])
    baselines = np.array([p.positive_fraction for p in pockets])
    by_protein: dict[str, list[float]] = {}
    for p in pockets:
        by_protein.setdefault(p.uniprot_id, []).append(p.auprc)
    multi = values[np.array([p.pockets_in_protein > 1 for p in pockets])]
    single = values[np.array([p.pockets_in_protein == 1 for p in pockets])]
    return {
        "auprc_pp": float(values.mean()),
        "auprc_pp_protein": float(np.mean([np.mean(v) for v in by_protein.values()])),
        "auprc_pp_multi": float(multi.mean()) if multi.size else float("nan"),
        "auprc_pp_single": float(single.mean()) if single.size else float("nan"),
        "auroc_pp": float(np.mean([p.auroc for p in pockets])),
        "r_precision_pp": float(np.mean([p.r_precision for p in pockets])),
        "hit1_pp": float(np.mean([p.hit_at_1 for p in pockets])),
        "lift": float(np.mean(values / baselines)),
        "n_pockets": len(pockets),
        "n_proteins": len(by_protein),
        "positive_fraction": float(baselines.mean()),
    }


def score_per_pocket(
    scores: dict[str, np.ndarray],
    dataset: PocketWatch,
    proteins: list[Protein],
    split: str,
) -> tuple[dict, list[PocketScore]]:
    """Per-pocket metrics for saved per-residue predictions.

    Args:
        scores: ``{uniprot_id: (L,) score vector}``.
        dataset: The benchmark, read for the site table and the site policy.
        proteins: The split's proteins, from `zpocket.data.load_split` — already carrying whatever
            multichain policy and protein filtering the run applied.
        split: Which split these are.

    Returns:
        ``(summary, per-pocket rows)``.
    """
    sites = load_sites(dataset, split)
    scored: list[PocketScore] = []
    missing_scores = missing_sites = 0
    for protein in proteins:
        if protein.uniprot_id not in scores:
            missing_scores += 1
            continue
        if protein.uniprot_id not in sites:
            missing_sites += 1
            continue
        scored.extend(
            score_one_protein(scores[protein.uniprot_id], protein, sites[protein.uniprot_id])
        )
    if missing_scores:
        logger.warning("%d proteins in the split have no saved prediction", missing_scores)
    if missing_sites:
        logger.warning("%d proteins have predictions but no sites row", missing_sites)
    return summarize_pockets(scored), scored
