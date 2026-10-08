"""Ground-truth discrete pockets: where the ligands actually sit, in the fold's frame.

The per-residue label table is a union over pockets; this is its decomposition. Each site carries
the canonical positions lining it — exactly the residues that make it positive per residue — and
the centroid of the ligands that defined it, projected into the fold frame.

Which sites a split scores is **not** decided here. It is read from the benchmark's
``site_policy.parquet``, the same artifact the per-residue half reads, which is the whole reason
the two halves score the same pockets. Re-deriving the decision from dates and chain annotations is
error-prone: done at two different granularities it disagrees in both directions on the same
pockets, which is why this reads the shipped artifact instead.

`load_partition` *masks* every unscored site rather than discarding it. A prediction landing on an
interface-only or already-public pocket has found a real pocket, so charging it as a false positive
is as wrong there as it would be on a pre-cutoff one — it is dropped from the ranked list instead.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from zpocket.data.pocketwatch import SCORED, SPLIT_CUTOFF, PocketWatch, _check_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Site:
    """One ground-truth pocket: its ligand centroid, and the residues lining it."""

    site_id: str
    uniprot_id: str
    centroid: np.ndarray  # (3,) ligand centroid in the fold frame
    positions: np.ndarray  # canonical positions lining the site
    first_observed: dt.date
    is_novel: bool  # first seen at or after this split's temporal cutoff
    member_count: int
    spread_angstrom: float  # furthest contributing ligand from the mean centroid


# The shipped `site_centroids.parquet` and a table rebuilt by `zpocket.discrete.centroids` use
# these names.
_CENTROID_COLUMNS = (
    "pocket_site_id",
    "normalized_uniprot_id",
    "split",
    "site_first_observed",
    "canonical_positions",
    "ligand_centroid_x",
    "ligand_centroid_y",
    "ligand_centroid_z",
    "centroid_member_count",
    "centroid_spread_angstrom",
)


def _site_from_row(row: dict, cutoff: dt.date | None) -> Site:
    return Site(
        site_id=row["pocket_site_id"],
        uniprot_id=row["normalized_uniprot_id"],
        centroid=np.array(
            [row["ligand_centroid_x"], row["ligand_centroid_y"], row["ligand_centroid_z"]],
            dtype=np.float32,
        ),
        positions=np.asarray(row["canonical_positions"], dtype=np.int32),
        first_observed=row["site_first_observed"],
        is_novel=cutoff is None or row["site_first_observed"] >= cutoff,
        member_count=row["centroid_member_count"],
        spread_angstrom=row["centroid_spread_angstrom"],
    )


def load_partition(
    dataset: PocketWatch,
    split: str,
    *,
    site_centroids: str | pl.DataFrame | None = None,
) -> tuple[dict[str, list[Site]], dict[str, list[Site]]]:
    """Scored and masked sites for one split, as the benchmark's site policy decides them.

    Args:
        dataset: The benchmark, read for `site_policy.parquet`.
        split: One of `zpocket.data.SPLITS`.
        site_centroids: Centroid table to read, as a path or an in-memory frame. None uses the
            benchmark's shipped `site_centroids.parquet`, which is in the frame of the folds in
            ``structures/`` — pass a table from `zpocket.discrete.centroids` to score against any
            other fold set, ESMFold2 included.

    Returns:
        ``(scored, masked)``, each ``{uniprot_id: [Site, ...]}``. Nothing is discarded: every site
        the policy does not score comes back as masked.

    Raises:
        ValueError: If `split` is unknown, or if the policy does not cover every site in the
            centroid table — a partial artifact would silently drop pockets from both halves.
    """
    _check_split(split)
    if site_centroids is None:
        table = dataset.site_centroids()
    elif isinstance(site_centroids, pl.DataFrame):
        table = site_centroids
    else:
        table = pl.read_parquet(site_centroids)
    table = table.filter(pl.col("split") == split)

    decisions = dict(
        dataset.site_policy()
        .filter(pl.col("split") == split)
        .select("pocket_site_id", "decision")
        .iter_rows()
    )
    absent = set(table["pocket_site_id"].to_list()) - set(decisions)
    if absent:
        raise ValueError(
            f"{split}: {len(absent)} sites in the centroid table are missing from the site "
            f"policy, e.g. {sorted(absent)[0]}"
        )

    cutoff = SPLIT_CUTOFF[split]
    scored: dict[str, list[Site]] = {}
    masked: dict[str, list[Site]] = {}
    for row in table.iter_rows(named=True):
        into = scored if decisions[row["pocket_site_id"]] == SCORED else masked
        into.setdefault(row["normalized_uniprot_id"], []).append(_site_from_row(row, cutoff))
    logger.info(
        "%s: %d sites scored over %d proteins, %d masked over %d proteins",
        split,
        sum(len(v) for v in scored.values()),
        len(scored),
        sum(len(v) for v in masked.values()),
        len(masked),
    )
    return scored, masked


def restrict(sites: dict[str, list[Site]], uniprot_ids) -> dict[str, list[Site]]:
    """Keep only the proteins in `uniprot_ids`, dropping those left with no sites.

    Args:
        sites: Sites keyed by UniProt.
        uniprot_ids: The proteins to keep — typically the ones a run actually predicted.

    Returns:
        The restricted mapping.
    """
    keep = set(uniprot_ids)
    return {uid: v for uid, v in sites.items() if uid in keep and v}
