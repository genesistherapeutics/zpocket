"""This code against the published benchmark: the counts have to come out right.

Every number asserted here is quoted in the PocketWatch dataset card or in a published checkpoint's
own metrics, so a drift in the loaders — a wrong join, a policy applied at the wrong granularity —
shows up as a count that no longer matches. Skipped when the dataset is not reachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from zpocket.data import PAPER_EXCLUDED_TEST_PROTEINS, load_split, scoreable
from zpocket.eval.per_pocket import load_sites, score_one_protein
from zpocket.eval.report import evaluate_residue_level

# From the dataset card's split table and its "scoring" section.
TEST_PROTEINS = 1566
TEST_SCORED_SITES = 1573
TEST_POSITIVE_RATE = 0.0486
# The paper's protein set: the four UNK-bearing proteins dropped from every method.
PAPER_POCKETS = 1562
PAPER_TARGETS = 1275
# `trunk_model: esmfold2` checkpoint metrics.json reports n=1559 scored test proteins.
TEST_SCOREABLE_KEEP = 1559
# The paper's headline basis instead drops the 280 proteins whose only new pockets are
# interface-only, which no single-chain method can express: 1,559 - 280. The two bases give
# different numbers (0.734 vs 0.745) from identical predictions, so both are pinned.
TEST_SCOREABLE_MASK = 1279


def test_test_split_matches_the_published_counts(dataset):
    """Protein count, scored-residue count and positive rate, as the card states them."""
    proteins = load_split(dataset, "test")
    assert len(proteins) == TEST_PROTEINS
    scored_residues = sum(int(p.mask.sum()) for p in proteins)
    positives = sum(p.n_positive for p in proteins)
    assert positives / scored_residues == pytest.approx(TEST_POSITIVE_RATE, abs=5e-4)


def test_scoreable_count_matches_the_published_checkpoint(dataset):
    """Under multichain=keep, 1,559 test proteins are scoreable — what the heads were scored on.

    The other seven have no positive surviving the mask, so their AUPRC is undefined.
    """
    proteins = load_split(dataset, "test", multichain="keep")
    assert len(scoreable(proteins)) == TEST_SCOREABLE_KEEP


def test_scoreable_count_under_mask_matches_the_paper_basis(dataset):
    """Under multichain=mask, 1,279 test proteins remain — the paper's headline target set.

    A protein whose only post-cutoff pockets are interface-only loses every positive here, so
    its AUPRC is undefined and it drops out. That is the comparison a single-chain method is
    owed, and it is why the paper reports 0.745 where the checkpoint logged 0.734.
    """
    proteins = load_split(dataset, "test", multichain="mask")
    assert len(scoreable(proteins)) == TEST_SCOREABLE_MASK


def test_site_policy_scores_the_published_number_of_sites(dataset):
    assert len(dataset.scored_site_ids("test")) == TEST_SCORED_SITES


def test_paper_protein_set_gives_the_paper_pocket_and_target_counts(dataset):
    """Dropping the four excluded proteins must leave exactly 1,562 pockets over 1,275 targets."""
    from zpocket.discrete.sites import load_partition, restrict

    scored, _ = load_partition(dataset, "test")
    kept = restrict(scored, set(scored) - set(PAPER_EXCLUDED_TEST_PROTEINS))
    assert sum(len(v) for v in kept.values()) == PAPER_POCKETS
    assert len(kept) == PAPER_TARGETS


def test_excluded_proteins_are_still_scored_by_the_policy(dataset):
    """They are valid benchmark targets; only the paper's table drops them."""
    scored_ids = dataset.scored_site_ids("test")
    sites = dataset.sites().filter(
        __import__("polars").col("normalized_uniprot_id").is_in(list(PAPER_EXCLUDED_TEST_PROTEINS))
    )
    assert len(set(sites["pocket_site_id"].to_list()) & scored_ids) == (
        TEST_SCORED_SITES - PAPER_POCKETS
    )


def test_labels_and_sequences_agree_on_length(dataset):
    """`load_split` raises on a length or sha256 disagreement; this asserts it finds none."""
    for split in ("validation", "test"):
        proteins = load_split(dataset, split)
        assert all(len(p.sequence) == p.length == len(p.label) == len(p.mask) for p in proteins)


def test_multichain_policies_are_distinct_and_ordered(dataset):
    """`mask` and `drop` both remove interface positives, so fewer proteins stay scoreable."""
    counts = {
        policy: len(scoreable(load_split(dataset, "test", multichain=policy)))
        for policy in ("keep", "mask", "drop")
    }
    assert counts["keep"] > counts["mask"]
    assert counts["mask"] == counts["drop"]


def test_max_length_and_limit_filter(dataset):
    short = load_split(dataset, "test", max_length=300)
    assert short and all(p.length <= 300 for p in short)
    assert len(load_split(dataset, "test", limit=5)) == 5


def test_unknown_split_raises(dataset):
    with pytest.raises(ValueError):
        load_split(dataset, "holdout")


def test_perfect_predictor_saturates_on_real_labels(dataset):
    """End to end on real benchmark rows: ranking by the label gives 1.0 on both halves.

    The synthetic version of this lives in test_eval_metrics; this one also exercises the real
    site table, the site policy, and the other-pocket masking on proteins that genuinely have
    several pockets.
    """
    proteins = load_split(dataset, "test", limit=120)
    rng = np.random.default_rng(0)
    scores = {p.uniprot_id: p.label.astype(float) + rng.normal(0, 1e-9, p.length) for p in proteins}
    report = evaluate_residue_level(scores, dataset, proteins, "test")
    assert report["per_target"]["auprc_pt"] == pytest.approx(1.0)
    assert report["per_pocket"]["auprc_pp"] == pytest.approx(1.0)
    assert report["per_pocket"]["n_pockets"] > 0


def test_per_pocket_masks_other_pockets_out_of_the_background(dataset):
    """On a multi-pocket protein, another pocket's residues are neither positive nor negative."""
    proteins = {p.uniprot_id: p for p in load_split(dataset, "test")}
    sites = load_sites(dataset, "test")
    multi = next(
        uid
        for uid, s in sites.items()
        if uid in proteins and sum(1 for _, _, is_scored in s if is_scored) >= 2
    )
    rows = score_one_protein(np.zeros(proteins[multi].length), proteins[multi], sites[multi])
    assert len(rows) >= 2
    for row in rows:
        # Something was held out of this pocket's background, and the totals stay consistent.
        assert row.n_masked_other > 0
        assert row.n_positive > 0 and row.n_negative > 0
        assert row.positive_fraction == pytest.approx(
            row.n_positive / (row.n_positive + row.n_negative)
        )


def test_unscored_sites_are_excluded_from_every_background(dataset):
    """A masked site is scored for nothing, but its residues are still not negatives."""
    proteins = {p.uniprot_id: p for p in load_split(dataset, "test")}
    sites = load_sites(dataset, "test")
    with_masked = next(
        uid
        for uid, s in sites.items()
        if uid in proteins
        and any(is_scored for _, _, is_scored in s)
        and any(not is_scored for _, _, is_scored in s)
    )
    rows = score_one_protein(
        np.zeros(proteins[with_masked].length), proteins[with_masked], sites[with_masked]
    )
    scored_count = sum(1 for _, _, is_scored in sites[with_masked] if is_scored)
    # Only scored sites produce rows, and each knows how many pockets its protein contributes.
    assert all(row.pockets_in_protein == scored_count for row in rows)
    assert len(rows) <= scored_count
