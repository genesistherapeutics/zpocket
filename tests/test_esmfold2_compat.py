"""Regression tests for the historical ESMFold2 semantics used by zPocket."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from torch import nn

import zpocket.esmfold2.input as input_module
from zpocket.esmfold2.compat import (
    _published_atom_attention_forward,
    published_atom_attention,
)


@dataclass(frozen=True)
class _ParsedMSA:
    deletions: object | None


def test_published_msa_parsing_discards_deletion_features(monkeypatch):
    deletion_counts = object()
    monkeypatch.setattr(
        input_module.MSA,
        "from_a3m",
        lambda *args, **kwargs: _ParsedMSA(deletions=deletion_counts),
    )

    modern = input_module.msa_from_local_a3m("input.a3m")
    published = input_module.msa_from_local_a3m("input.a3m", zero_deletions=True)

    assert modern.deletions is deletion_counts
    assert published.deletions is None


def test_published_attention_patch_is_model_scoped_and_restored():
    from esm.models.esmfold2.layers import SWA3DRoPEAttention

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = SWA3DRoPEAttention(d_model=8, n_heads=2)

    model = _Model()
    unrelated = _Model()
    installed_forward = model.attention.forward.__func__
    unrelated_forward = unrelated.attention.forward.__func__

    with pytest.raises(RuntimeError, match="probe"), published_atom_attention(model):
        assert model.attention.forward.__func__ is _published_atom_attention_forward
        assert unrelated.attention.forward.__func__ is unrelated_forward
        raise RuntimeError("probe")

    assert model.attention.forward.__func__ is installed_forward


def test_fold_defaults_wire_all_published_compatibility_behaviors(monkeypatch, tmp_path):
    from esm.models.esmfold2.layers import SWA3DRoPEAttention

    import zpocket.esmfold2.fold as fold_module
    from zpocket.esmfold2.config import ESMFold2Config

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = SWA3DRoPEAttention(d_model=8, n_heads=2)

    class _Complex:
        @staticmethod
        def to_mmcif():
            return "data_test\n"

    class _Builder:
        def fold(self, model, spi, **kwargs):
            assert model.attention.forward.__func__ is _published_atom_attention_forward
            assert kwargs["msa_max_depth"] == 16384
            assert kwargs["msa_column_mask_rate"] == 0.0
            return SimpleNamespace(complex=_Complex())

    model = _Model()
    installed_forward = model.attention.forward.__func__
    real_build_spi = fold_module.build_spi
    seen = {}

    def _build_spi(*args, **kwargs):
        seen.update(kwargs)
        return real_build_spi(*args, **kwargs)

    monkeypatch.setattr(fold_module, "build_spi", _build_spi)
    fold_module.fold_protein(
        model,
        _Builder(),
        sequence="ACDE",
        structure_id="test",
        out_dir=str(tmp_path),
        cfg=ESMFold2Config(num_diffusion_samples=1),
        model_weight_uri="weights",
    )

    assert seen["zero_msa_deletions"] is True
    assert model.attention.forward.__func__ is installed_forward
