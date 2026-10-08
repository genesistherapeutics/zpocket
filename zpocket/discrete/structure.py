"""The folded structure behind a protein, in the one coordinate frame everything shares.

Every geometric quantity in this package — a predicted pocket centre, a ground-truth ligand
centroid — lives in the frame of one fold. That is the frame the pair features were computed in,
and the frame the ground-truth centroids are projected into, so no superposition is ever needed
downstream and none is ever applied.

Folds are single-chain monomers numbered 1..L over the canonical sequence, which makes a residue
number a canonical position directly. `load_fold` checks that rather than assuming it: a fold
numbered any other way would silently misalign every per-residue score against its coordinates.

This reads local paths only — plain or gzipped mmCIF, which gemmi opens transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One fold's heavy-atom coordinates, indexed by canonical position.

    `atom_position` and `atom_coord` are parallel arrays over heavy atoms, so a set of canonical
    positions selects its atoms with a single boolean test. `ca_coord` is the per-residue CA
    coordinate, NaN where the fold has no CA for that position.
    """

    uniprot_id: str
    length: int
    atom_position: np.ndarray  # (A,) int32, one-based canonical position
    atom_coord: np.ndarray  # (A, 3) float32
    ca_coord: np.ndarray  # (L, 3) float32, NaN where absent

    def atoms_of(self, positions) -> np.ndarray:
        """Heavy-atom coordinates of the given canonical positions.

        Args:
            positions: One-based canonical positions, e.g. a pocket site's lining residues.

        Returns:
            An ``(n, 3)`` array of the coordinates of every heavy atom in those residues.
        """
        wanted = np.zeros(self.length + 1, dtype=bool)
        inside = np.asarray(list(positions), dtype=int)
        wanted[inside[(inside >= 1) & (inside <= self.length)]] = True
        return self.atom_coord[wanted[self.atom_position]]

    def centroid_of(self, positions) -> np.ndarray:
        """Heavy-atom centroid of the given canonical positions, or NaN if none are present."""
        atoms = self.atoms_of(positions)
        return atoms.mean(axis=0) if len(atoms) else np.full(3, np.nan, dtype=np.float32)

    def ca_at(self, positions) -> tuple[np.ndarray, np.ndarray]:
        """CA coordinates for the given canonical positions, dropping those the fold lacks.

        Args:
            positions: One-based canonical positions.

        Returns:
            ``(positions_found, coords)`` — the subset of `positions` that lie inside the fold and
            have a CA, in the given order, and their ``(n, 3)`` coordinates. Used to build the
            correspondence a superposition needs, which is why the surviving positions come back
            alongside the coordinates rather than being silently dropped.
        """
        wanted = np.asarray(list(positions), dtype=np.int64)
        inside = wanted[(wanted >= 1) & (wanted <= self.length)]
        coords = self.ca_coord[inside - 1]
        present = ~np.isnan(coords[:, 0])
        return inside[present], coords[present]


def load_fold(path: str | Path, uniprot_id: str, expected_length: int) -> Fold:
    """Read a fold's heavy atoms, checking its numbering really is 1..L.

    Args:
        path: Local path to the fold, plain or gzipped mmCIF/PDB.
        uniprot_id: The protein this fold belongs to, carried through for provenance.
        expected_length: Canonical sequence length the per-residue scores are indexed against.

    Returns:
        The fold's heavy-atom coordinates and per-residue CA coordinates.

    Raises:
        ValueError: If the fold is not a single chain numbered within 1..`expected_length`. Any
            other numbering would misalign every per-residue score against its coordinates.
    """
    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    if len(structure) == 0 or len(structure[0]) != 1:
        raise ValueError(
            f"{uniprot_id}: expected a single-chain monomer, got "
            f"{0 if len(structure) == 0 else len(structure[0])} chains in {path}"
        )

    positions: list[int] = []
    coords: list[tuple[float, float, float]] = []
    ca = np.full((expected_length, 3), np.nan, dtype=np.float32)
    for residue in structure[0][0]:
        number = int(residue.seqid.num)
        if not 1 <= number <= expected_length:
            raise ValueError(
                f"{uniprot_id}: residue number {number} outside 1..{expected_length}; the fold is "
                f"not numbered against the canonical sequence, so scores cannot be aligned to it"
            )
        for atom in residue:
            if atom.element == gemmi.Element("H"):
                continue
            positions.append(number)
            coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
            if atom.name == "CA":
                ca[number - 1] = (atom.pos.x, atom.pos.y, atom.pos.z)

    return Fold(
        uniprot_id=uniprot_id,
        length=expected_length,
        atom_position=np.asarray(positions, dtype=np.int32),
        atom_coord=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
        ca_coord=ca,
    )


# Accepted fold file names, in preference order. A gzipped mmCIF comes first since that is how the
# benchmark ships them; the `_sample<k>` forms are what `zpocket run` and `zpocket fold` write, so
# their output directory can be handed straight to `centroids` and `eval` with no renaming. Sample
# 0 is the conformer whose pair features were captured, so it is the only one scored.
_FOLD_SUFFIXES = (
    ".cif.gz",
    ".cif",
    "_sample0.cif",
    "_sample0.cif.gz",
    ".pdb.gz",
    ".pdb",
)


def find_fold_file(directory: str | Path, uniprot_id: str) -> Path | None:
    """Locate one protein's fold in a directory, trying the usual names.

    Args:
        directory: Directory of folds named by UniProt accession, either ``<id>.cif[.gz]`` or
            the ``<id>_sample0.cif`` that `zpocket run` / `zpocket fold` write.
        uniprot_id: The accession to look for.

    Returns:
        The path, or None if no candidate exists.
    """
    for suffix in _FOLD_SUFFIXES:
        candidate = Path(directory) / f"{uniprot_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def make_fold_resolver(
    lengths: dict[str, int],
    *,
    structures_dir: str | Path | None = None,
    dataset=None,
    cache: bool = True,
):
    """Build a ``uniprot_id -> Fold | None`` callable over a directory of folds.

    Args:
        lengths: Canonical sequence length per UniProt — the number `load_fold` validates the
            residue numbering against, so this must come from the benchmark's own sequences.
        structures_dir: Directory of folds named ``<uniprot_id>.cif[.gz]``. Pass the ESMFold2
            folds here.
        dataset: A `zpocket.data.PocketWatch` to fall back on, which serves the release's own
            ``structures/``. Pair it with that release's ``site_centroids.parquet`` and with no
            other centroid table.
        cache: Hold loaded folds in memory. A split's folds are a few hundred megabytes of
            coordinates and are read twice (once to decode, once to build targets).

    Returns:
        A callable returning the fold, or None when the protein has no fold or no known length.

    Raises:
        ValueError: If neither `structures_dir` nor `dataset` is given — there would be nowhere
            to read a fold from.
    """
    if structures_dir is None and dataset is None:
        raise ValueError("pass structures_dir, a dataset, or both")
    loaded: dict[str, Fold | None] = {}

    def resolve(uniprot_id: str) -> Fold | None:
        if cache and uniprot_id in loaded:
            return loaded[uniprot_id]
        fold = None
        length = lengths.get(uniprot_id)
        if length is not None:
            path = (
                find_fold_file(structures_dir, uniprot_id) if structures_dir is not None else None
            )
            if path is None and dataset is not None:
                try:
                    path = dataset.structure_path(uniprot_id)
                except Exception:  # noqa: BLE001 - absent fold is a normal outcome, not an error
                    path = None
            if path is not None:
                fold = load_fold(path, uniprot_id, length)
        if cache:
            loaded[uniprot_id] = fold
        return fold

    return resolve
