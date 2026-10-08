"""Shared fixtures: a benchmark handle, skipped when the dataset is not reachable.

The metric tests split into two kinds. Those that assert a property of a metric build their own
arrays and need nothing external. Those that assert a *count* — 1,566 test proteins, 1,573 scored
sites — are checking this code against the published benchmark, so they need it present, and are
skipped rather than failed when it is absent (no network, no cache).

A skip passes, so a run that is meant to vouch for the counts sets ``POCKETWATCH_REQUIRED=1``,
which turns an unreachable dataset or missing folds into a failure.
"""

from __future__ import annotations

import os
from typing import NoReturn

import pytest


def _unavailable(reason: str) -> NoReturn:
    """Skip, or fail when ``POCKETWATCH_REQUIRED`` says the dataset has to be there."""
    if os.environ.get("POCKETWATCH_REQUIRED"):
        pytest.fail(f"{reason} (POCKETWATCH_REQUIRED is set)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def dataset():
    """A `PocketWatch` handle, or a skip when its tables cannot be reached.

    Honours ``POCKETWATCH_ROOT`` for a local dataset tree; otherwise resolves from the Hub, which
    needs either a warm HuggingFace cache or network access.
    """
    pw = pytest.importorskip("zpocket.data").PocketWatch(os.environ.get("POCKETWATCH_ROOT"))
    try:
        pw.labels()
    except Exception as exc:  # noqa: BLE001 - any failure to reach the data is a skip, not a bug
        _unavailable(f"PocketWatch not reachable: {type(exc).__name__}: {exc}")
    return pw


@pytest.fixture(scope="session")
def shipped_folds(dataset):
    """Skip unless the shipped ESMFold2 folds are cached locally.

    One fold is enough to tell a warm cache from a cold one; the centroid test needs many, and
    downloading them inside a test run is not something a test should do.

    The probe has to be a validation or test protein: ``structures/`` holds only those two splits
    (train's folds ship as ``structures_train.zip``), so probing the first protein in the
    *sequences* table picks a train protein and reports a 404 as "no folds".
    """
    probe = dataset.site_centroids()["normalized_uniprot_id"][0]
    try:
        path = dataset.structure_path(probe)
    except Exception as exc:  # noqa: BLE001
        _unavailable(f"shipped folds not available: {type(exc).__name__}: {exc}")
    if not os.path.exists(path):
        _unavailable("shipped folds not cached")
    return dataset
