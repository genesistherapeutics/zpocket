"""The sampling recipe actually reaching ``ESMFold2InputBuilder.fold``.

`ESMFold2Config` defaults to the published recipe, but a default is only worth anything if it
is forwarded. `_fold_kwargs` filters the recipe to the parameters the installed ``esm``
accepts, so these tests pin both halves of that: every knob is passed to a builder that takes
them, and a builder that does not take one is reported rather than silently obeyed.

`num_loops` and `lm_dropout` are the load-bearing pair — both recompute `z`, so dropping
either changes the prediction, not just the output structure.
"""

from __future__ import annotations

import argparse
import logging

from zpocket.cli import _recipe_overrides
from zpocket.esmfold2.config import (
    PAPER_SAMPLING,
    PUBLISHED_MSA_COLUMN_MASK_RATE,
    PUBLISHED_MSA_MAX_DEPTH,
    ESMFold2Config,
)
from zpocket.esmfold2.fold import _fold_kwargs


class _FullBuilder:
    """A builder whose `fold` takes every knob the published recipe sets."""

    def fold(
        self,
        model,
        input,
        *,
        num_loops=20,
        num_sampling_steps=200,
        num_diffusion_samples=1,
        seed=None,
        noise_scale=None,
        step_scale=None,
        max_inference_sigma=None,
        lm_dropout=0.3,
        msa_max_depth=1024,
        msa_column_mask_rate=0.1,
        complex_id="pred",
    ):
        raise AssertionError("not called")


class _OldBuilder:
    """An older builder with no `lm_dropout` and no MSA-subsampling knobs."""

    def fold(
        self,
        model,
        input,
        *,
        num_loops=3,
        num_sampling_steps=200,
        num_diffusion_samples=1,
        seed=None,
        noise_scale=None,
        step_scale=None,
        max_inference_sigma=None,
        complex_id="pred",
    ):
        raise AssertionError("not called")


class _KwargsBuilder:
    """A builder taking ``**kwargs``, where nothing can be known to be unsupported."""

    def fold(self, model, input, **kwargs):
        raise AssertionError("not called")


def test_published_recipe_reaches_a_builder_that_accepts_it():
    """Every entry of PAPER_SAMPLING is forwarded, at its published value."""
    kwargs = _fold_kwargs(_FullBuilder(), ESMFold2Config(), "sid")
    for name, value in PAPER_SAMPLING.items():
        assert kwargs[name] == value, f"{name} not forwarded as {value}"
    assert kwargs["msa_max_depth"] == PUBLISHED_MSA_MAX_DEPTH
    assert kwargs["msa_column_mask_rate"] == PUBLISHED_MSA_COLUMN_MASK_RATE
    assert kwargs["complex_id"] == "sid"


def test_modern_esm_semantics_leave_msa_knobs_unset():
    """The explicit opt-out delegates subsampling and masking to installed esm."""
    cfg = ESMFold2Config(published_checkpoint_compatibility=False)
    assert cfg.msa_max_depth is None and cfg.msa_column_mask_rate is None
    kwargs = _fold_kwargs(_FullBuilder(), cfg, "sid")
    assert "msa_max_depth" not in kwargs
    assert "msa_column_mask_rate" not in kwargs


def test_modern_esm_cli_flag_disables_published_compatibility():
    overrides = _recipe_overrides(argparse.Namespace(modern_esm_semantics=True))
    assert overrides["published_checkpoint_compatibility"] is False


def test_explicitly_set_msa_knobs_are_forwarded():
    cfg = ESMFold2Config(msa_max_depth=16384, msa_column_mask_rate=0.0)
    kwargs = _fold_kwargs(_FullBuilder(), cfg, "sid")
    assert kwargs["msa_max_depth"] == 16384
    # 0.0 is a meaningful value, not "unset": it must survive the None check.
    assert kwargs["msa_column_mask_rate"] == 0.0


def test_unsupported_knobs_are_dropped_and_warned_about(caplog):
    """An esm too old for `lm_dropout` must not silently produce different features."""
    with caplog.at_level(logging.WARNING):
        kwargs = _fold_kwargs(_OldBuilder(), ESMFold2Config(lm_dropout=0.25), "sid")
    assert "lm_dropout" not in kwargs
    # The knobs it does support still arrive, so the fold is not abandoned.
    assert kwargs["num_loops"] == 10
    assert any("lm_dropout" in r.getMessage() for r in caplog.records), (
        "dropping a knob that changes z must be reported"
    )


def test_a_kwargs_builder_receives_everything():
    """With **kwargs there is nothing to filter against, so pass the whole recipe."""
    kwargs = _fold_kwargs(_KwargsBuilder(), ESMFold2Config(), "sid")
    assert kwargs["num_loops"] == 10 and kwargs["lm_dropout"] == 0.25
