"""The discrete half: the Kabsch projection, the cluster decoder, and the DCC metrics.

All of this runs on folds built here, so none of it needs the dataset. The projection test is the
important one — it is the operation that puts the localization ground truth into a new fold's
frame, which is how the benchmark is scored against folds it does not ship.
"""

from __future__ import annotations

import numpy as np
import pytest

from zpocket.discrete.decode import cluster, mean_shift_modes, probabilities
from zpocket.discrete.metrics import (
    SiteTarget,
    average_precision,
    distances_to_site,
    evaluate,
    greedy_matches,
)
from zpocket.discrete.project import kabsch, project
from zpocket.discrete.sites import Site
from zpocket.discrete.structure import Fold


def _fold(n_residues: int = 30, spacing: float = 6.0) -> Fold:
    """A fold whose residues sit on a line, one heavy atom (the CA) each.

    A line keeps every distance analytic, so an expected centroid can be written down.
    """
    positions = np.arange(1, n_residues + 1, dtype=np.int32)
    coords = np.column_stack(
        [positions.astype(np.float32) * spacing, np.zeros(n_residues), np.zeros(n_residues)]
    ).astype(np.float32)
    return Fold(
        uniprot_id="TEST",
        length=n_residues,
        atom_position=positions,
        atom_coord=coords,
        ca_coord=coords.copy(),
    )


def _random_rigid(seed: int = 0):
    """A random rotation and translation, for checking the projection recovers them."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.normal(scale=10.0, size=3)


def test_kabsch_recovers_a_known_rigid_transform():
    """Superposing transformed points onto the originals returns the inverse transform exactly."""
    rng = np.random.default_rng(1)
    source = rng.normal(size=(12, 3))
    rotation, translation = _random_rigid(2)
    target = source @ rotation.T + translation

    fitted_rotation, fitted_translation = kabsch(source, target)
    assert fitted_rotation == pytest.approx(rotation, abs=1e-9)
    assert fitted_translation == pytest.approx(translation, abs=1e-9)


def test_kabsch_never_returns_a_reflection():
    """A reflection superposes mirrored points just as well but turns a structure inside out."""
    rng = np.random.default_rng(3)
    source = rng.normal(size=(20, 3))
    target = source * np.array([1.0, 1.0, -1.0])  # a pure reflection
    rotation, _ = kabsch(source, target)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_project_carries_a_ligand_into_the_fold_frame():
    """A ligand centroid moves with the superposition that lands its contacts on the fold."""
    rng = np.random.default_rng(4)
    fold_ca = rng.normal(scale=5.0, size=(8, 3))
    centroid_in_fold = fold_ca.mean(axis=0) + np.array([1.0, 2.0, -1.5])

    # Build an "entry" by applying a known rigid transform to both the CAs and the ligand.
    rotation, translation = _random_rigid(5)
    entry_ca = (fold_ca - translation) @ rotation
    entry_centroid = (centroid_in_fold - translation) @ rotation

    result = project(entry_ca, fold_ca, entry_centroid)
    assert result is not None
    assert result.fit_rmsd_angstrom == pytest.approx(0.0, abs=1e-8)
    assert result.centroid == pytest.approx(centroid_in_fold, abs=1e-8)
    assert result.matched_ca_count == 8


def test_project_refuses_an_underdetermined_fit():
    """Two matched CAs leave a rotation free in one axis, so there is no transform to apply."""
    assert project(np.zeros((2, 3)), np.ones((2, 3)), np.zeros(3)) is None


def test_cluster_separates_two_distant_pockets():
    """Two groups of confident residues far apart must decode to two centres, not one."""
    fold = _fold(n_residues=40, spacing=6.0)
    scores = np.full(40, 0.01)
    scores[2:7] = 0.95  # one pocket near the start
    scores[30:35] = 0.90  # another near the end, far beyond link_distance
    candidates = cluster(
        scores, fold, method="linkage", score_threshold=0.5, link_distance=8.0, min_residues=3
    )
    assert len(candidates) == 2
    # Ranked by summed score, so the higher-scoring group comes first.
    assert candidates[0].score >= candidates[1].score
    assert candidates[0].rank == 1 and candidates[1].rank == 2
    centres = np.stack([c.center for c in candidates])
    assert np.linalg.norm(centres[0] - centres[1]) > 100.0


def test_cluster_chains_through_a_bridge_but_meanshift_does_not():
    """Single-linkage merges two pockets joined by confident residues; mean shift does not.

    This is the documented difference between the two methods, and the reason both exist.

    The assertion is made on the mode labels rather than on `cluster`'s output because mean
    shift *over*-fragments at this bandwidth rather than finding exactly two modes: these
    residues sit on a uniformly spaced line, which has no density gradient for a flat kernel to
    climb, so seeds barely move and adjacent pairs end up as their own modes. That is why
    `min_residues` would discard every one of them.

    Note the bandwidth here is 8 A, below the ~14 A the published decoder uses. That is the
    point: mean shift is sensitive to a bandwidth narrower than a pocket's residue shell, and
    this test pins the failure mode rather than endorsing the setting. At the published
    bandwidth `meanshift` is the better decoder and is `zpocket eval`'s default.
    """
    fold = _fold(n_residues=40, spacing=4.0)
    scores = np.full(40, 0.01)
    scores[5:12] = 0.95
    scores[18:25] = 0.95
    scores[12:18] = 0.6  # the bridge: confident enough to be selected, and it links the two
    linkage = cluster(scores, fold, score_threshold=0.5, link_distance=8.0, method="linkage")
    assert len(linkage) == 1

    selected = np.flatnonzero(scores >= 0.5)
    labels = mean_shift_modes(fold.ca_coord[selected], scores[selected], 8.0)
    assert labels.max() + 1 > 1, "mean shift must not collapse the bridge into one mode"

    # With the size filter lifted, those modes do surface as separate candidates.
    shift = cluster(
        scores, fold, score_threshold=0.5, method="meanshift", bandwidth=8.0, min_residues=1
    )
    assert len(shift) >= 2


def test_cluster_always_emits_something():
    """With nothing above threshold it falls back to the top residues, rather than skipping."""
    fold = _fold()
    candidates = cluster(
        np.full(fold.length, 0.01), fold, method="linkage", score_threshold=0.5
    )
    assert len(candidates) >= 1


def test_cluster_rejects_a_mismatched_score_vector():
    """One score per residue, or the scores are not aligned to the coordinates at all."""
    with pytest.raises(ValueError):
        cluster(np.zeros(5), _fold(n_residues=30))
    with pytest.raises(ValueError):
        cluster(np.zeros(30), _fold(n_residues=30), method="nonsense")


def test_mean_shift_merges_seeds_in_one_basin():
    """Points in a single tight cluster all climb to the same mode."""
    points = np.random.default_rng(6).normal(scale=0.5, size=(15, 3))
    labels = mean_shift_modes(points, np.ones(15), bandwidth=8.0)
    assert len(set(labels.tolist())) == 1


def test_probabilities_is_a_sigmoid():
    assert probabilities(np.array([0.0])) == pytest.approx([0.5])
    assert probabilities(np.array([-50.0, 50.0])) == pytest.approx([0.0, 1.0], abs=1e-9)


def _target(centroid, positions=(1, 2, 3)) -> SiteTarget:
    return SiteTarget(
        site=Site(
            site_id="s",
            uniprot_id="TEST",
            centroid=np.asarray(centroid, dtype=np.float32),
            positions=np.asarray(positions, dtype=np.int32),
            first_observed=None,
            is_novel=True,
            member_count=1,
            spread_angstrom=0.0,
        ),
        lining_atoms=np.asarray([centroid], dtype=np.float32),
    )


def test_distances_to_site_measures_from_each_prediction():
    target = _target([0.0, 0.0, 0.0])
    dcc, dca = distances_to_site(np.array([[3.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), target)
    assert dcc == pytest.approx([3.0, 10.0])
    assert dca == pytest.approx([3.0, 10.0])


def test_greedy_matching_is_one_to_one():
    """A single centre between two sites claims one of them, not both.

    Precision needs this: counting it twice would credit two true positives for one prediction.
    """
    targets = [_target([0.0, 0.0, 0.0]), _target([2.0, 0.0, 0.0])]
    matches = greedy_matches(np.array([[1.0, 0.0, 0.0]]), targets, threshold=4.0)
    assert matches.tolist() == [True]
    assert int(matches.sum()) == 1


def test_top_n_budget_uses_independent_matching():
    """A site is checked against every kept prediction, as the published top-N protocol does.

    So one centre sitting between two nearby sites is a hit for both here, even though the
    precision-side matching above counts it once.
    """
    targets = {"TEST": [_target([0.0, 0.0, 0.0]), _target([2.0, 0.0, 0.0])]}
    result = evaluate({"TEST": np.array([[1.0, 0.0, 0.0]])}, targets, threshold=4.0, extra=(0, 2))
    assert result["dcc_top_n"] == pytest.approx(1.0)
    assert result["dcc_top_n_plus_2"] == pytest.approx(1.0)
    assert result["n_sites"] == 2


def test_extra_budget_can_only_help():
    """top-(N+2) reads two more predictions than top-N, so it is never the lower number."""
    targets = {"TEST": [_target([0.0, 0.0, 0.0])]}
    # The true site is the third-ranked prediction: outside top-N=1, inside top-N+2=3.
    centres = np.array([[50.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result = evaluate({"TEST": centres}, targets, threshold=4.0, extra=(0, 2))
    assert result["dcc_top_n"] == pytest.approx(0.0)
    assert result["dcc_top_n_plus_2"] == pytest.approx(1.0)
    assert result["dcc_ceiling"] == pytest.approx(1.0)


def test_a_protein_with_no_predictions_counts_as_a_failure():
    """A detector that proposes nothing must not be rewarded with an absent row."""
    targets = {"TEST": [_target([0.0, 0.0, 0.0])]}
    result = evaluate({}, targets, threshold=4.0)
    assert result["dcc_top_n"] == pytest.approx(0.0)
    assert result["n_proteins"] == 1
    assert result["mean_predictions"] == pytest.approx(0.0)


def test_masked_sites_are_neither_credited_nor_charged():
    """A prediction landing only on a masked pocket is dropped, not counted as a false positive.

    Without the drop, finding a real pocket the split excludes would lower precision — penalising
    a method for being right.
    """
    scored = {"TEST": [_target([0.0, 0.0, 0.0])]}
    masked = {"TEST": [_target([100.0, 0.0, 0.0])]}
    centres = np.array([[0.5, 0.0, 0.0], [100.2, 0.0, 0.0]])

    with_mask = evaluate({"TEST": centres}, scored, threshold=4.0, masked=masked)
    without = evaluate({"TEST": centres}, scored, threshold=4.0)
    assert with_mask["dcc_top_n"] == pytest.approx(1.0)
    # The masked-site prediction is gone, so precision sees one prediction, not two.
    assert with_mask["precision_pooled"] == pytest.approx(1.0)
    assert without["precision_pooled"] == pytest.approx(0.5)


def test_average_precision_charges_for_over_emitting():
    """AP falls when unmatched predictions outrank the true positive; a top-N rate cannot see that."""
    targets = [_target([0.0, 0.0, 0.0])]
    hit_first = average_precision(np.array([[0.1, 0.0, 0.0], [99.0, 0.0, 0.0]]), targets, 4.0)
    hit_second = average_precision(np.array([[99.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), targets, 4.0)
    assert hit_first == pytest.approx(1.0)
    assert hit_second == pytest.approx(0.5)
    assert np.isnan(average_precision(np.zeros((1, 3)), [], 4.0))
