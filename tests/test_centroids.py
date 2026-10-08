"""The centroid rebuild, checked against the table the benchmark ships.

This is the test that licenses scoring the localization task on folds the benchmark does not ship.
`site_centroids.parquet` is in the frame of the folds in ``structures/``; running the rebuild
over those same folds must reproduce it. If it does, then running it over a fold set you generated
yourself produces the same ground truth in *that* frame, and its localization numbers mean
something.

The dataset card puts the reference agreement at 1e-5 A, which is the tolerance asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from zpocket.discrete.centroids import build_site_centroids
from zpocket.discrete.structure import make_fold_resolver

TOLERANCE_ANGSTROM = 1e-5
N_SAMPLE_PROTEINS = 40


def _resolver(dataset):
    lengths = dict(
        dataset.sequences().select("normalized_uniprot_id", "sequence_length").iter_rows()
    )
    return make_fold_resolver(lengths, dataset=dataset)


def test_rebuild_reproduces_the_shipped_centroids(shipped_folds):
    """Project into the shipped folds and the shipped centroids come back, to 1e-5 A.

    Scoped to a sample of proteins: the full table is 4,466 sites over 3,119 folds and takes
    minutes, while the operation under test is per observation and identical at any scale.
    """
    dataset = shipped_folds
    shipped = dataset.site_centroids()
    sample = sorted(set(shipped["normalized_uniprot_id"].to_list()))[:N_SAMPLE_PROTEINS]

    built = build_site_centroids(dataset, _resolver(dataset), only=sample)
    assert built.height > 0

    joined = built.join(shipped, on="pocket_site_id", how="inner", suffix="_ref")
    assert joined.height == built.height, "every rebuilt site must exist in the shipped table"

    deviation = np.linalg.norm(
        np.column_stack(
            [
                joined["ligand_centroid_x"] - joined["ligand_centroid_x_ref"],
                joined["ligand_centroid_y"] - joined["ligand_centroid_y_ref"],
                joined["ligand_centroid_z"] - joined["ligand_centroid_z_ref"],
            ]
        ),
        axis=1,
    )
    assert deviation.max() < TOLERANCE_ANGSTROM, f"max deviation {deviation.max():.3e} A"
    # The member count is how many observations were projected; a mismatch means this dropped
    # or gained observations relative to the reference, even where the mean happens to agree.
    assert (joined["centroid_member_count"] == joined["centroid_member_count_ref"]).all()


def test_rebuilt_table_drives_the_partition(shipped_folds):
    """A rebuilt table is accepted by `load_partition` in place of the shipped one.

    This is the seam the ESMFold2 path depends on, so it is worth asserting directly rather than
    only through the CLI.
    """
    from zpocket.discrete.sites import load_partition

    dataset = shipped_folds
    shipped = dataset.site_centroids()
    sample = sorted(set(shipped["normalized_uniprot_id"].to_list()))[:N_SAMPLE_PROTEINS]
    built = build_site_centroids(dataset, _resolver(dataset), only=sample)

    scored, masked = load_partition(dataset, "test", site_centroids=built)
    # Every returned site belongs to the sample, and the two halves do not overlap.
    assert set(scored) | set(masked) <= set(sample)
    assert not (set(scored) & set(masked)) or all(
        {s.site_id for s in scored[uid]} & {s.site_id for s in masked[uid]} == set()
        for uid in set(scored) & set(masked)
    )


def test_partition_rejects_a_table_the_policy_does_not_cover(shipped_folds):
    """An unknown site id must fail loudly: a partial policy would silently drop pockets."""
    import polars as pl

    from zpocket.discrete.sites import load_partition

    dataset = shipped_folds
    rogue = (
        dataset.site_centroids()
        .filter(pl.col("split") == "test")
        .head(1)
        .with_columns(pl.lit("not-a-real-site-id").alias("pocket_site_id"))
    )
    with pytest.raises(ValueError, match="missing from the site policy"):
        load_partition(dataset, "test", site_centroids=rogue)


def test_fold_numbering_is_validated(shipped_folds):
    """A fold whose residues do not lie in 1..L is rejected rather than silently misaligned."""
    from zpocket.discrete.structure import load_fold

    import polars as pl

    dataset = shipped_folds
    # A validation/test protein: only those two splits have folds under `structures/`.
    uid = dataset.site_centroids()["normalized_uniprot_id"][0]
    length = int(
        dataset.sequences().filter(pl.col("normalized_uniprot_id") == uid)["sequence_length"][0]
    )
    # Declaring the fold far shorter than it is puts its own residue numbers out of range.
    with pytest.raises(ValueError, match="outside 1.."):
        load_fold(dataset.structure_path(uid), uid, expected_length=5)

    fold = load_fold(dataset.structure_path(uid), uid, length)
    assert fold.length == length
    assert fold.atom_position.min() >= 1 and fold.atom_position.max() <= length


def test_find_fold_file_accepts_zpocket_run_output(tmp_path):
    """`zpocket run` writes <id>_sample0.cif, so its output dir needs no renaming downstream.

    Sample 0 is the conformer whose pair features were captured, so it is the one to score.
    """
    from zpocket.discrete.structure import find_fold_file

    assert find_fold_file(tmp_path, "P12345") is None
    (tmp_path / "P12345_sample0.cif").write_text("data_x\n")
    assert find_fold_file(tmp_path, "P12345").name == "P12345_sample0.cif"
    # A plain <id>.cif wins when both exist -- it is the benchmark's own spelling.
    (tmp_path / "P12345.cif").write_text("data_x\n")
    assert find_fold_file(tmp_path, "P12345").name == "P12345.cif"
