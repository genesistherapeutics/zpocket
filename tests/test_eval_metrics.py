"""Properties the per-residue metrics must have, checked on arrays built here.

These need no dataset and no weights. They pin the two things a reimplementation gets wrong: the
scale (a perfect ranker is 1.0, a random one is the positive fraction) and the mask contract (a
masked residue must not influence any number).
"""

from __future__ import annotations

import numpy as np
import pytest

from zpocket.data import Protein
from zpocket.eval.metrics import (
    hit_at_1,
    pocket_auprc,
    pocket_auroc,
    r_precision,
    summarize,
)
from zpocket.eval.per_pocket import score_one_protein


def _toy(n_proteins: int = 12, length: int = 80, rate: float = 0.1, seed: int = 0):
    """Labels with `rate` positives per protein, and a matching all-ones mask."""
    rng = np.random.default_rng(seed)
    labels, masks = [], []
    for _ in range(n_proteins):
        label = (rng.random(length) < rate).astype(np.uint8)
        label[0] = 1  # guarantee both classes, so every protein is scoreable
        label[1] = 0
        labels.append(label)
        masks.append(np.ones(length, dtype=np.uint8))
    return labels, masks


def test_perfect_ranker_scores_one():
    """Ranking by the label itself is a perfect ranker: every metric saturates."""
    labels, masks = _toy()
    scores = [label.astype(float) for label in labels]
    result = summarize(scores, labels, masks)
    assert result["auprc_pt"] == pytest.approx(1.0)
    assert result["auprc_pool"] == pytest.approx(1.0)
    assert result["auroc_pt"] == pytest.approx(1.0)
    assert result["r_precision_pt"] == pytest.approx(1.0)
    assert result["hit1_pt"] == pytest.approx(1.0)


def test_inverted_ranker_is_worse_than_random():
    """Ranking by the negated label puts every positive last — AUROC 0, not 0.5."""
    labels, masks = _toy()
    scores = [-label.astype(float) for label in labels]
    assert pocket_auroc(scores, labels, masks) == pytest.approx(0.0)


def test_random_ranker_lands_near_the_positive_fraction():
    """A random ranker's AUPRC is its prevalence; that is the baseline every number is read against."""
    rng = np.random.default_rng(1)
    labels, masks = _toy(n_proteins=400, length=400, rate=0.05, seed=2)
    scores = [rng.random(len(label)) for label in labels]
    result = pocket_auprc(scores, labels, masks, mode="pooled")
    assert result.auprc == pytest.approx(result.positive_fraction, abs=0.01)


def test_masked_residues_cannot_influence_the_score():
    """A masked residue is excluded entirely: corrupting one must not move any metric.

    This is the contract that a reimplementation most often breaks by scoring a masked residue as
    a 0, which turns an unknown into a negative and inflates the result.
    """
    labels, masks = _toy(seed=3)
    scores = [label.astype(float) for label in labels]
    for mask in masks:
        mask[10:20] = 0

    clean = summarize(scores, labels, masks)
    corrupted = [s.copy() for s in scores]
    for s in corrupted:
        s[10:20] = 999.0  # rank the masked residues top; they must still not count
    assert summarize(corrupted, labels, masks) == pytest.approx(clean, nan_ok=True)


def test_per_target_is_not_pooled():
    """Per-target weights proteins equally, so a long protein cannot dominate a short one.

    One easy short protein and one hard long one: the per-target mean sits between the two
    per-protein values, while pooling is dragged toward the long one.
    """
    short_label = np.array([1, 1, 0, 0], dtype=np.uint8)
    long_label = np.concatenate([np.ones(2, dtype=np.uint8), np.zeros(998, dtype=np.uint8)])
    scores = [np.array([1.0, 0.9, 0.1, 0.0]), np.concatenate([[0.0, 0.0], np.ones(998)])]
    labels = [short_label, long_label]
    per_target = pocket_auprc(scores, labels, mode="per_target")
    pooled = pocket_auprc(scores, labels, mode="pooled")
    assert per_target.auprc > pooled.auprc
    assert per_target.n_targets_scored == 2


def test_r_precision_and_hit_at_1_read_the_top_of_the_ranking():
    """`r_precision` is precision in the top-k (k = positive count); `hit_at_1` is the top residue."""
    labels = np.array([1, 1, 0, 0, 0])
    assert r_precision(np.array([0.9, 0.8, 0.1, 0.1, 0.1]), labels) == pytest.approx(1.0)
    assert r_precision(np.array([0.1, 0.1, 0.9, 0.8, 0.7]), labels) == pytest.approx(0.0)
    assert r_precision(np.array([0.9, 0.1, 0.8, 0.1, 0.1]), labels) == pytest.approx(0.5)
    assert hit_at_1(np.array([0.9, 0.1, 0.8, 0.1, 0.1]), labels) == pytest.approx(1.0)
    assert hit_at_1(np.array([0.1, 0.1, 0.9, 0.1, 0.1]), labels) == pytest.approx(0.0)


def test_unscoreable_targets_are_skipped_not_zeroed():
    """A protein with no positives has undefined AUPRC and must not be counted as a 0."""
    labels = [np.array([1, 0, 0, 0], dtype=np.uint8), np.zeros(4, dtype=np.uint8)]
    scores = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5, 0.5])]
    result = pocket_auprc(scores, labels, mode="per_target")
    assert result.n_targets == 2
    assert result.n_targets_scored == 1
    assert result.auprc == pytest.approx(1.0)


def test_per_pocket_keeps_unscored_sites_out_of_the_background():
    """An unscored site's residues are neither positive nor negative for any scored pocket.

    Built so that nothing else protects them: label 0 and inside the mask, which is how the
    ``drop`` policy leaves an interface-only site. Ranking them top must not move the AUPRC.
    """
    length = 40
    label = np.zeros(length, dtype=np.uint8)
    label[0:5] = 1
    protein = Protein("P00000", length, "A" * length, label, np.ones(length, dtype=np.uint8))
    sites = [("scored", np.arange(0, 5), True), ("unscored", np.arange(10, 15), False)]
    scores = label.astype(float)
    scores[10:15] = 2.0

    (row,) = score_one_protein(scores, protein, sites)
    assert row.pocket_site_id == "scored"
    assert row.auprc == pytest.approx(1.0)
    assert row.n_negative == length - 10
    assert row.n_masked_other == 5


def test_mismatched_shapes_raise():
    """A score vector that is not one value per residue is a misalignment, not a warning."""
    with pytest.raises(ValueError):
        pocket_auprc([np.zeros(5)], [np.zeros(6, dtype=np.uint8)])
    with pytest.raises(ValueError):
        pocket_auprc([np.zeros(5)], [np.zeros(5, dtype=np.uint8)], mode="nonsense")
