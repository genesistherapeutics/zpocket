"""Tests for the plain-torch ZPairformerHead pocket head.

The tests that need the trained weights skip gracefully when they are absent; the round-trip and
refusal tests need no external files. The real-weight tests assert that strict-load succeeds with
zero missing / unexpected keys and ``n_params == 330177``, which is what licenses calling this a
port rather than a reimplementation.

The published head is the ``genesisml/zPocket`` release: ``model.safetensors`` + ``config.json``.
Set ``ZPOCKET_HEADS_DIR`` to additionally exercise local ``.pt`` training checkpoints laid out as
``<dir>/*/*/pairformer2_seed0.pt``, which is the other format ``load_pairformer_head`` accepts.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

from zpocket.heads.pairformer import ZPairformerHead, load_pairformer_head
from zpocket.predict import predict_pockets

_EXPECTED_N_PARAMS = 330177


def _real_ckpts() -> list[str]:
    """Resolve local ``.pt`` training checkpoints, if any are staged.

    Returns an empty list when ``ZPOCKET_HEADS_DIR`` is unset or empty, which turns the
    ``.pt``-format tests into skips rather than errors. The published release ships
    ``model.safetensors`` instead; it is covered by the release tests further down.
    """
    local = os.environ.get("ZPOCKET_HEADS_DIR")
    if not local:
        return []
    return sorted(glob.glob(os.path.join(local, "*", "*", "pairformer2_seed0.pt")))


@pytest.mark.parametrize("ckpt", _real_ckpts() or [pytest.param(None, marks=pytest.mark.skip)])
def test_strict_load_real_ckpt(ckpt: str | None) -> None:
    """Each real checkpoint strict-loads with zero missing/unexpected keys and n_params == 330177."""
    if ckpt is None or not os.path.exists(ckpt):
        pytest.skip("no local .pt checkpoints staged (set ZPOCKET_HEADS_DIR)")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["head"] == "pairformer2"
    pair_dim = int(ck["pair_dim"])

    # Direct strict load to capture missing/unexpected explicitly (must both be empty).
    head = ZPairformerHead(pair_dim, n_pairformer=2)
    result = head.load_state_dict(ck["state_dict"], strict=True)
    assert result.missing_keys == [], f"missing keys: {result.missing_keys}"
    assert result.unexpected_keys == [], f"unexpected keys: {result.unexpected_keys}"

    # The public loader must also succeed and report the expected param count.
    loaded, meta = load_pairformer_head(ckpt)
    assert meta["n_params"] == _EXPECTED_N_PARAMS
    assert sum(p.numel() for p in loaded.parameters()) == _EXPECTED_N_PARAMS
    assert not loaded.training  # .eval() applied


@pytest.mark.parametrize("ckpt", _real_ckpts() or [pytest.param(None, marks=pytest.mark.skip)])
def test_forward_shape_real_ckpt(ckpt: str | None) -> None:
    """Loaded head runs on random z (48, 48, pair_dim) -> finite (48,) logits; probs in [0, 1]."""
    if ckpt is None or not os.path.exists(ckpt):
        pytest.skip("no local .pt checkpoints staged (set ZPOCKET_HEADS_DIR)")

    head, meta = load_pairformer_head(ckpt)
    L, c = 48, meta["pair_dim"]
    z = np.random.default_rng(0).standard_normal((L, L, c)).astype(np.float32)

    with torch.no_grad():
        logits = head(torch.from_numpy(z))
    assert logits.shape == (L,)
    assert torch.isfinite(logits).all()

    probs = predict_pockets(z, head)
    assert probs.shape == (L,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert np.isfinite(probs).all()


def test_state_dict_round_trip(tmp_path) -> None:
    """Build a head, save its state_dict, reload strict with zero missing/unexpected keys."""
    torch.manual_seed(0)
    head = ZPairformerHead(pair_dim=256, n_pairformer=2)
    assert sum(p.numel() for p in head.parameters()) == _EXPECTED_N_PARAMS

    path = tmp_path / "rt.pt"
    torch.save({"state_dict": head.state_dict(), "head": "pairformer2", "pair_dim": 256}, path)

    reloaded = ZPairformerHead(256, n_pairformer=2)
    result = reloaded.load_state_dict(
        torch.load(path, weights_only=False)["state_dict"], strict=True
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    # Public loader path also round-trips.
    loaded, meta = load_pairformer_head(str(path))
    assert meta["n_params"] == _EXPECTED_N_PARAMS

    # Weights actually match (same forward on the same input).
    L = 16
    z = torch.randn(L, L, 256)
    with torch.no_grad():
        assert torch.allclose(head.eval()(z), loaded(z), atol=1e-6)


# --- the published weights-only release format ------------------------------------------------

SAFETENSORS_REPO = os.environ.get("ZPOCKET_SAFETENSORS_REPO", "genesisml/zPocket")


def _safetensors_release():
    """Download `model.safetensors` + its `config.json`, or None if unreachable."""
    try:
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(SAFETENSORS_REPO, "model.safetensors")
        hf_hub_download(SAFETENSORS_REPO, "config.json")  # must land beside the weights
        return weights
    except Exception:  # noqa: BLE001 - no network/auth/repo is a skip, not a failure
        return None


def test_safetensors_release_loads_strictly():
    """The weights-only release loads with no missing or unexpected keys.

    The release ships `model.safetensors` + `config.json` and **no torch pickle**, so the
    architecture is read from the config. This asserts the port still matches the published
    export exactly — 52 tensors, 330,177 parameters.
    """
    weights = _safetensors_release()
    if weights is None:
        pytest.skip(f"{SAFETENSORS_REPO} not reachable")

    head, meta = load_pairformer_head(weights)
    assert meta["head"] == "pairformer2"
    assert meta["pair_dim"] == 256
    assert meta["n_params"] == _EXPECTED_N_PARAMS
    assert meta["source"] == "safetensors"
    assert not head.training

    # Strict load again, explicitly, so missing/unexpected keys are asserted rather than implied.
    from safetensors.torch import load_file

    fresh = ZPairformerHead(256, n_pairformer=2)
    result = fresh.load_state_dict(load_file(weights), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert len(load_file(weights)) == 52


def test_safetensors_without_a_config_is_refused(tmp_path):
    """No config.json means no architecture; guessing one could load into the wrong shape."""
    from safetensors.torch import save_file

    orphan = tmp_path / "model.safetensors"
    save_file({"proj.weight": torch.zeros(64, 256)}, str(orphan))
    with pytest.raises(FileNotFoundError, match="no config.json"):
        load_pairformer_head(str(orphan))
