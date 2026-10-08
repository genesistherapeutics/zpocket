"""The PocketWatch benchmark: per-residue labels, discrete sites, and which of them are scored.

The benchmark ships as the HuggingFace dataset ``genesisml/PocketWatch``. `PocketWatch` resolves
a file either from a local directory (a clone, or an extracted download) or from the Hub on
demand, so nothing here needs the network once a file is cached.

Three contracts carry over from the benchmark and are the ones worth knowing:

``mask`` **is the scoring contract.** ``mask[i] == 0`` means residue ``i`` is neither a positive
nor a negative and must be excluded from metrics. Scoring it as a 0 corrupts them. Validation and
test mask out residues of pockets first seen before the split's cutoff, so each split scores only
the pockets that were new to it.

``site_policy.parquet`` **decides which pockets count**, and it is a property of the benchmark
rather than of a model. A site it does not score is still a real pocket: never a negative, and its
residues stay out of every other pocket's background. Both the per-residue and the per-centroid
half read this one artifact, which is what makes the two comparable pocket for pocket — so neither
re-derives the decision from dates or chain annotations.

There is no index of pre-computed features: the benchmark ships sequences, and trunk features
are something you generate. `load_split` therefore takes the sequences table as its join partner
and leaves feature/structure paths to the caller.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
import os
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# The published release. `POCKETWATCH_HF_REPO` overrides it, which is what pointing at a staging
# mirror or a fork needs; `PocketWatch(root=...)` bypasses the Hub entirely.
HF_REPO_ID = os.environ.get("POCKETWATCH_HF_REPO", "genesisml/PocketWatch")
SPLITS = ("train", "validation", "test")

# The temporal boundaries the splits were cut on. A site is novel to its split when first observed
# on or after that split's own cutoff; train has no cutoff, so every train site is novel to it.
SPLIT_CUTOFF: dict[str, dt.date | None] = {
    "train": None,
    "validation": dt.date(2021, 9, 30),
    "test": dt.date(2024, 1, 1),
}

# Interface-only contacts exist only where two chains meet, so a monomer fold cannot support them.
# "keep" scores them as stored; "mask" excludes them from scoring (the fair comparison against a
# single-chain method); "drop" calls them negatives. This is a *residue*-level policy and is not
# the pocket-level exclusion in `site_policy.parquet` — the two disagree in both directions on the
# same pockets, which is exactly why the benchmark ships the pocket-level decision as an artifact.
MULTICHAIN_POLICIES = ("keep", "mask", "drop")

# The paper drops these four test proteins from every method: their sequences carry non-standard
# residues (written UNK in the folds) that the VN-EGNN baseline cannot read, so keeping them would
# put a column on a different protein set from its neighbours. They remain valid benchmark targets
# and `site_policy.parquet` scores them — drop them only to compare against the paper's table.
PAPER_EXCLUDED_TEST_PROTEINS = ("A0A655NVH2", "P03441", "Q7XJE8", "Q962Y6")

SCORED = "scored"  # the two `decision` values in site_policy.parquet
MASKED = "masked"

_LABELS = "data/pocket_labels.parquet"
_SITES = "data/pocket_label_sites.parquet"
_SEQUENCES = "data/sequences.parquet"
_SITE_POLICY = "data/site_policy.parquet"
_SITE_CENTROIDS = "data/site_centroids.parquet"
_LIGAND_CENTROIDS = "projection/ligand_centroids.parquet"
_CONTACT_RESIDUES = "projection/contact_residues.parquet"


class PocketWatch:
    """Locates PocketWatch files, from a local directory or the Hub.

    A local `root` is used as-is and nothing is downloaded, which is what an offline or
    air-gapped run needs. Without one, files are pulled from `HF_REPO_ID` through the normal
    HuggingFace cache, so each is fetched once per machine.
    """

    def __init__(self, root: str | None = None, *, repo_id: str = HF_REPO_ID) -> None:
        """
        Args:
            root: Directory holding the dataset tree (``data/``, ``projection/``, ``structures/``).
                None resolves every file from the Hub instead.
            repo_id: Hub dataset to fall back on when `root` is None.
        """
        self.root = root
        self.repo_id = repo_id

    def path(self, relpath: str) -> str:
        """Local path to one dataset file, downloading it if this is a Hub-backed instance.

        Args:
            relpath: Path within the dataset, e.g. ``"data/pocket_labels.parquet"``.

        Returns:
            An existing local path.

        Raises:
            FileNotFoundError: If a local `root` was given and does not hold the file. Falling
                back to the Hub here would silently mix a local dataset with a published one,
                and the two can be different revisions.
        """
        if self.root is not None:
            candidate = os.path.join(self.root, relpath)
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"{candidate} not found. Either the dataset root is incomplete, or omit "
                    f"`root` to resolve {relpath!r} from {self.repo_id} on the Hub."
                )
            return candidate
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.repo_id, relpath, repo_type="dataset")

    # Each table is read once per process: `load_split` and the per-pocket scorer want the same
    # label frame, and it is 14k rows of dense per-residue arrays.
    @functools.lru_cache(maxsize=None)
    def _read(self, relpath: str) -> pl.DataFrame:
        return pl.read_parquet(self.path(relpath))

    def labels(self) -> pl.DataFrame:
        """Per-UniProt label table: dense ``label`` / ``mask`` / ``single_chain_reachable``."""
        return self._read(_LABELS)

    def sites(self) -> pl.DataFrame:
        """Per-site table: the canonical positions lining each pocket, and its dates."""
        return self._read(_SITES)

    def sequences(self) -> pl.DataFrame:
        """Canonical sequence per UniProt, with its sha256 — what the folds are numbered against."""
        return self._read(_SEQUENCES)

    def site_policy(self) -> pl.DataFrame:
        """Which sites are scored and why. The authority for both halves of the benchmark."""
        return self._read(_SITE_POLICY)

    def site_centroids(self) -> pl.DataFrame:
        """Ligand centroids in the sample-0 frame of this release's own ``structures/``.

        The published release folds with ESMFold2 and expresses these centroids in that frame,
        so they pair directly with ``structures/``. Earlier revisions shipped a different fold
        set and centroids in that frame; the file name is the same either way, so a table and a fold set
        from different revisions silently disagree. Against any fold set other than the one in
        the same revision, rebuild with `zpocket.discrete.centroids` from `ligand_centroids`
        and `contact_residues`.
        """
        return self._read(_SITE_CENTROIDS)

    def ligand_centroids(self) -> pl.DataFrame:
        """Per-observation ligand centroids in their own deposited-entry frames."""
        return self._read(_LIGAND_CENTROIDS)

    def contact_residues(self) -> pl.DataFrame:
        """Per-observation entry-frame CA coordinates, tagged with canonical position."""
        return self._read(_CONTACT_RESIDUES)

    def structure_path(self, uniprot_id: str) -> str:
        """Path to one shipped fold (gzipped mmCIF), numbered 1..L."""
        return self.path(f"structures/{uniprot_id}.cif.gz")

    def scored_site_ids(self, split: str) -> set[str]:
        """Site ids the benchmark scores for one split.

        Args:
            split: One of `SPLITS`.

        Returns:
            The ``pocket_site_id`` values whose ``decision`` is ``scored``.
        """
        _check_split(split)
        frame = self.site_policy().filter(
            (pl.col("split") == split) & (pl.col("decision") == SCORED)
        )
        return set(frame["pocket_site_id"].to_list())


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


@dataclass(frozen=True)
class Protein:
    """One labelled protein: its sequence and per-residue targets.

    `label` and `mask` are indexed by ``position - 1`` over the canonical sequence, which is the
    same numbering the folds use, so a score vector, a label and a fold residue all line up
    without any remapping.
    """

    uniprot_id: str
    length: int
    sequence: str
    label: np.ndarray  # (L,) uint8
    mask: np.ndarray  # (L,) uint8 — 1 = counts, 0 = excluded entirely

    @property
    def n_positive(self) -> int:
        """Positives that survive the mask — a protein with none is unscoreable."""
        return int((self.label & self.mask).sum())


def apply_multichain_policy(
    label: np.ndarray, mask: np.ndarray, single_chain_reachable: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve interface-only contacts — positives a monomer fold cannot support.

    Args:
        label: (L,) 0/1 per-residue pocket labels.
        mask: (L,) 0/1 scoring mask; 0 means the residue counts as neither positive nor negative.
        single_chain_reachable: (L,) 0/1 flag for contacts a single chain can express.
        policy: ``"keep"`` scores interface contacts as stored, ``"mask"`` excludes them from
            scoring, ``"drop"`` turns them into negatives.

    Returns:
        The ``(label, mask)`` pair after applying the policy.

    Raises:
        ValueError: If `policy` is not one of `MULTICHAIN_POLICIES`.
    """
    if policy == "keep":
        return label, mask
    if policy == "mask":
        return label, mask & (single_chain_reachable | (1 - label))
    if policy == "drop":
        return label & single_chain_reachable, mask
    raise ValueError(f"unknown multichain policy {policy!r}; expected one of {MULTICHAIN_POLICIES}")


def load_split(
    dataset: PocketWatch,
    split: str,
    *,
    multichain: str = "keep",
    max_length: int | None = None,
    limit: int | None = None,
    only: Collection[str] | None = None,
    exclude: Collection[str] = (),
) -> list[Protein]:
    """Load one split's labelled proteins.

    Args:
        dataset: The benchmark to read from.
        split: One of `SPLITS`.
        multichain: How interface-only contacts count; see `apply_multichain_policy`. This changes
            which proteins are scoreable, so numbers under two policies are not comparable.
        max_length: Drop proteins longer than this. None keeps every length, which is what the
            published head checkpoints were trained and scored at.
        limit: Keep only the first this-many proteins, for smoke runs.
        only: Restrict to these UniProt ids.
        exclude: Drop these UniProt ids. Pass `PAPER_EXCLUDED_TEST_PROTEINS` to match the paper.

    Returns:
        Proteins in UniProt-id order, each carrying its labels and mask.

    Raises:
        ValueError: If `split` is unknown, or if a protein's label row and its sequence disagree
            on length or sha256 — which would misalign every residue.
    """
    _check_split(split)
    labels = dataset.labels().filter(pl.col("split") == split)
    sequences = dataset.sequences().select(
        "normalized_uniprot_id", "sequence", "sequence_length", "sequence_sha256"
    )
    joined = labels.join(sequences, on="normalized_uniprot_id", how="inner", suffix="_seq")
    if joined.height < labels.height:
        raise ValueError(
            f"{split}: {labels.height - joined.height} labelled proteins have no sequence row; "
            f"the dataset tree is inconsistent"
        )
    disagree = joined.filter(
        (pl.col("sequence_length") != pl.col("sequence_length_seq"))
        | (pl.col("sequence_sha256") != pl.col("sequence_sha256_seq"))
    )
    if disagree.height:
        raise ValueError(
            f"{split}: {disagree.height} proteins disagree with their sequence on length or "
            f"sha256, e.g. {disagree['normalized_uniprot_id'][0]}"
        )

    if only is not None:
        joined = joined.filter(pl.col("normalized_uniprot_id").is_in(list(only)))
    if exclude:
        joined = joined.filter(~pl.col("normalized_uniprot_id").is_in(list(exclude)))
    if max_length is not None:
        joined = joined.filter(pl.col("sequence_length") <= max_length)
    joined = joined.sort("normalized_uniprot_id")
    if limit is not None:
        joined = joined.head(limit)

    proteins = []
    for row in joined.iter_rows(named=True):
        label = np.asarray(row["label"], dtype=np.uint8)
        mask = np.asarray(row["mask"], dtype=np.uint8)
        reachable = np.asarray(row["single_chain_reachable"], dtype=np.uint8)
        label, mask = apply_multichain_policy(label, mask, reachable, multichain)
        proteins.append(
            Protein(
                uniprot_id=row["normalized_uniprot_id"],
                length=row["sequence_length"],
                sequence=row["sequence"],
                label=label,
                mask=mask,
            )
        )
    scored_residues = sum(int(p.mask.sum()) for p in proteins)
    logger.info(
        "%s: %d proteins, %d residues scored, %d positives (%.2f%%), multichain=%s",
        split,
        len(proteins),
        scored_residues,
        sum(p.n_positive for p in proteins),
        100 * sum(p.n_positive for p in proteins) / max(1, scored_residues),
        multichain,
    )
    return proteins


def scoreable(proteins: list[Protein]) -> list[Protein]:
    """Drop proteins with no surviving positives — AUPRC is undefined for them.

    A multichain transform can leave a validation or test protein with zero positives when its
    only post-cutoff pocket was an interface pocket. They are excluded from evaluation here,
    never from the label file.

    Args:
        proteins: Proteins as returned by `load_split`.

    Returns:
        The subset with at least one positive surviving the mask.
    """
    keep = [p for p in proteins if p.n_positive > 0]
    if len(keep) < len(proteins):
        logger.info(
            "dropping %d proteins with no positives from evaluation", len(proteins) - len(keep)
        )
    return keep
