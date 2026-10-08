"""Rebuild the localization ground truth in *your* fold's frame.

A centroid metric needs the ligand's position in the same frame as the predictions, and the two
live apart: the ligand is deposited in its PDB entry's frame, the predictions in the fold's. The
benchmark's ``projection/`` tables hold both halves of the bridge — each observation's contact-residue
CA coordinates in its entry frame, tagged with the canonical position each maps to, and that
observation's ligand centroid in the same entry frame. Superposing those CA atoms onto the fold's
CA atoms at the same canonical positions gives a rigid transform, and applying it to the centroid
lands the ligand where the predictions are.

The fit is local, on the contact residues alone rather than the whole chain: a pocket is rigid even
where the rest of the protein is not, and a whole-chain fit would spend its error budget on loops
far from the site.

This is why the benchmark can be scored on a fold set it does not ship. ``site_centroids.parquet``
is expressed in the frame of its own release's ``structures/`` and means nothing against any other
fold set, so a run with a different seed, a different ``num_loops``, or a different model needs its
own table; `build_site_centroids` produces one. Run over the fold set the shipped centroids were
built from, it reproduces the shipped table, which is how this implementation is tested.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable

import numpy as np
import polars as pl

from zpocket.data.pocketwatch import PocketWatch
from zpocket.discrete.project import MIN_MATCHED_CA, project
from zpocket.discrete.structure import Fold

logger = logging.getLogger(__name__)

# Columns `zpocket.discrete.sites.load_partition` reads, in the shipped table's own spelling.
OUTPUT_COLUMNS = (
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
    "median_fit_rmsd_angstrom",
)


def project_observation(
    contacts: np.ndarray, entry_centroid: np.ndarray, fold: Fold
) -> tuple[np.ndarray, float] | None:
    """Carry one observation's ligand centroid into a fold's frame.

    Args:
        contacts: ``(n, 4)`` rows of ``(canonical_position, ca_x, ca_y, ca_z)`` in the entry frame.
        entry_centroid: ``(3,)`` ligand centroid in the same entry frame.
        fold: The fold to land the ligand in.

    Returns:
        ``(centroid, fit_rmsd_angstrom)`` in the fold frame, or None when fewer than
        `MIN_MATCHED_CA` positions are shared with the fold — too few to determine a transform,
        and guessing one would place the ligand arbitrarily.
    """
    positions = contacts[:, 0].astype(np.int64)
    found, fold_ca = fold.ca_at(positions)
    if len(found) < MIN_MATCHED_CA:
        return None
    # `ca_at` keeps the order it was given and drops what the fold lacks, so re-select the entry
    # rows by the surviving positions to keep the two sides in correspondence.
    index = {int(p): i for i, p in enumerate(positions)}
    entry_ca = contacts[[index[int(p)] for p in found], 1:4].astype(np.float64)
    result = project(entry_ca, fold_ca.astype(np.float64), entry_centroid)
    if result is None:
        return None
    return result.centroid, result.fit_rmsd_angstrom


def build_site_centroids(
    dataset: PocketWatch,
    fold_for: Callable[[str], Fold | None],
    *,
    splits: Iterable[str] = ("validation", "test"),
    only: Collection[str] | None = None,
) -> pl.DataFrame:
    """Project every site's ligand centroid into the frame of the folds `fold_for` supplies.

    Each site's centroid is the mean over its member observations' projected centroids. They agree
    closely — the members are the same pocket seen in different entries — and `centroid_spread_angstrom`
    reports the disagreement so a caller can drop a site whose members do not converge.

    Args:
        dataset: The benchmark, read for the projection tables and per-site metadata.
        fold_for: Resolves a UniProt id to its fold, or None when that protein has no fold.
            Proteins it returns None for are skipped and counted, never silently zeroed.
        splits: Which splits to build. The projection bundle covers validation and test, the two
            splits the localization task is defined on.
        only: Restrict to these UniProt ids. Use it to rebuild a handful of proteins rather than a
            whole split — a full split reads 889k contact rows and every fold.

    Returns:
        One row per site that got at least one projected member, with `OUTPUT_COLUMNS`. Feed it
        to `zpocket.discrete.sites.load_partition` as `site_centroids`.
    """
    splits = tuple(splits)
    meta = (
        dataset.sites()
        .filter(pl.col("split").is_in(list(splits)))
        .select(
            "pocket_site_id",
            "normalized_uniprot_id",
            "split",
            "site_first_observed",
            "canonical_positions",
        )
    )
    if only is not None:
        meta = meta.filter(pl.col("normalized_uniprot_id").is_in(list(only)))
    observations = dataset.ligand_centroids().join(
        meta.select("pocket_site_id"), on="pocket_site_id", how="semi"
    )
    contacts = dataset.contact_residues().join(
        observations.select("pocket_observation_id"), on="pocket_observation_id", how="semi"
    )
    # One (n, 4) block per observation, built once: the alternative is a filter per observation
    # over 889k rows.
    by_observation = {
        row["pocket_observation_id"]: np.column_stack(
            [row["canonical_position"], row["ca_x"], row["ca_y"], row["ca_z"]]
        )
        for row in contacts.group_by("pocket_observation_id")
        .agg(pl.col("canonical_position"), pl.col("ca_x"), pl.col("ca_y"), pl.col("ca_z"))
        .iter_rows(named=True)
    }

    folds: dict[str, Fold | None] = {}
    projected: dict[str, list[tuple[np.ndarray, float]]] = {}
    no_fold: set[str] = set()
    no_contacts = too_few = 0
    for row in observations.iter_rows(named=True):
        uid = row["normalized_uniprot_id"]
        if uid not in folds:
            folds[uid] = fold_for(uid)
        fold = folds[uid]
        if fold is None:
            no_fold.add(uid)
            continue
        block = by_observation.get(row["pocket_observation_id"])
        if block is None or not len(block):
            no_contacts += 1
            continue
        result = project_observation(
            block,
            np.array(
                [row["ligand_centroid_x"], row["ligand_centroid_y"], row["ligand_centroid_z"]],
                dtype=np.float64,
            ),
            fold,
        )
        if result is None:
            too_few += 1
            continue
        projected.setdefault(row["pocket_site_id"], []).append(result)

    rows = []
    for site in meta.iter_rows(named=True):
        members = projected.get(site["pocket_site_id"])
        if not members:
            continue
        centroids = np.stack([c for c, _ in members])
        mean = centroids.mean(axis=0)
        rows.append(
            {
                "pocket_site_id": site["pocket_site_id"],
                "normalized_uniprot_id": site["normalized_uniprot_id"],
                "split": site["split"],
                "site_first_observed": site["site_first_observed"],
                "canonical_positions": site["canonical_positions"],
                "ligand_centroid_x": float(mean[0]),
                "ligand_centroid_y": float(mean[1]),
                "ligand_centroid_z": float(mean[2]),
                "centroid_member_count": len(members),
                "centroid_spread_angstrom": float(np.linalg.norm(centroids - mean, axis=1).max()),
                "median_fit_rmsd_angstrom": float(np.median([r for _, r in members])),
            }
        )
    if no_fold:
        logger.warning("%d proteins have no fold; their sites are absent", len(no_fold))
    if no_contacts or too_few:
        logger.warning(
            "dropped %d observations with no contact rows and %d with fewer than %d matched CAs",
            no_contacts,
            too_few,
            MIN_MATCHED_CA,
        )
    logger.info("built %d of %d sites over splits %s", len(rows), meta.height, list(splits))
    return pl.DataFrame(rows, schema_overrides={"site_first_observed": pl.Date})
