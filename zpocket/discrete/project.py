"""Put a ligand where the fold can see it: superpose its entry onto the fold, and carry it across.

A centroid metric needs the ligand's position in the same frame as the predictions, and the two
live apart — the ligand is deposited in its PDB entry's frame, the predictions in the fold's. The
annotation tables hold both halves of the bridge: the entry's CA coordinates for the residues the
ligand contacts, each tagged with the canonical position it maps to, and the ligand's centroid in
that same entry frame. Superposing those CA atoms onto the fold's CA atoms at the same canonical
positions gives a rigid transform, and applying it to the centroid lands the ligand in the fold.

The fit is local, on the contact residues alone rather than the whole chain. A pocket is rigid
even where the rest of the protein is not, and a whole-chain fit would spend its error budget on
loops far from the site.

`fit_rmsd_angstrom` is the honest quality signal: a pocket whose residues do not superpose is one
whose ligand position should not be trusted, and the caller can drop it on that basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Below three matched CA atoms a rigid transform is underdetermined in at least one axis.
MIN_MATCHED_CA = 3


@dataclass(frozen=True)
class Projection:
    """One ligand carried into the fold's frame, with the evidence for trusting it."""

    centroid: np.ndarray  # (3,) in the fold frame
    matched_ca_count: int
    fit_rmsd_angstrom: float


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation carrying `source` onto `target` with least squared error.

    Args:
        source: ``(n, 3)`` points to move.
        target: ``(n, 3)`` points to move them onto, in correspondence with `source`.

    Returns:
        ``(rotation, translation)`` such that ``source @ rotation.T + translation`` best matches
        `target`.

    Raises:
        ValueError: If the two arrays are not the same ``(n, 3)`` shape.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected matching (n, 3) arrays, got {source.shape} and {target.shape}")

    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    # Flip the least significant axis when the naive solution is a reflection, which superposes
    # the points just as well but turns the structure inside out.
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - rotation @ source_centre


def project(
    source_ca: np.ndarray, fold_ca: np.ndarray, source_centroid: np.ndarray
) -> Projection | None:
    """Carry one ligand centroid from its entry's frame into the fold's.

    Args:
        source_ca: ``(n, 3)`` CA coordinates from the deposited entry.
        fold_ca: ``(n, 3)`` CA coordinates of the same canonical positions in the fold.
        source_centroid: ``(3,)`` ligand centroid in the entry's frame.

    Returns:
        The projection, or None when fewer than `MIN_MATCHED_CA` positions matched — too few to
        determine a transform, and guessing one would place the ligand arbitrarily.
    """
    if len(source_ca) < MIN_MATCHED_CA:
        return None
    rotation, translation = kabsch(source_ca, fold_ca)
    fitted = source_ca @ rotation.T + translation
    rmsd = float(np.sqrt(((fitted - fold_ca) ** 2).sum(axis=1).mean()))
    return Projection(
        centroid=(rotation @ np.asarray(source_centroid, dtype=np.float64) + translation),
        matched_ca_count=len(source_ca),
        fit_rmsd_angstrom=rmsd,
    )
