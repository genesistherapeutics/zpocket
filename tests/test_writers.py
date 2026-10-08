"""Round-trip tests for both trunk-artifact writers (numpy-only, no GPU/model)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from zpocket.esmfold2.trunk_artifacts import (
    TRUNK_ARTIFACT_SCHEMA_VERSION,
    write_trunk_artifact,
)

SEQ = "MKTAYIAKQR"
N = len(SEQ)
C_S = 8
C_Z = 4
N_BINS = 6


def _esm_input() -> dict:
    return {"sequences": [{"type": "protein", "id": "A", "sequence": SEQ}]}


def _load(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# ESMFold2 writer
# --------------------------------------------------------------------------- #


def test_esmfold2_s_only(tmp_path):
    path = tmp_path / "s.npz"
    s = np.random.randn(N, C_S).astype(np.float32)
    write_trunk_artifact(
        path,
        esmfold2_input=_esm_input(),
        structure_id="prot1",
        structure_seed=7,
        model_weight_uri="local://model",
        s=s,
    )
    loaded = _load(path)
    assert loaded["s"].shape == (N, C_S)
    assert loaded["s"].dtype == np.float16
    assert "z" not in loaded and "pdistogram" not in loaded and "esmc" not in loaded
    assert int(loaded["schema_version"]) == TRUNK_ARTIFACT_SCHEMA_VERSION
    assert int(loaded["structure_seed"]) == 7
    assert str(loaded["sequence_sha256"]) == hashlib.sha256(SEQ.encode()).hexdigest()
    assert str(loaded["model_name"]) == "ESMFold2"


def test_esmfold2_z_and_pdistogram(tmp_path):
    path = tmp_path / "z.npz"
    z = np.random.randn(N, N, C_Z).astype(np.float32)
    pdist = np.random.randn(N, N, N_BINS).astype(np.float32)
    boundaries = np.linspace(2.0, 22.0, N_BINS - 1).astype(np.float32)
    write_trunk_artifact(
        path,
        esmfold2_input=_esm_input(),
        structure_id="prot1",
        structure_seed=0,
        model_weight_uri="local://model",
        z=z,
        pdistogram=pdist,
        distogram_bin_boundaries_angstrom=boundaries,
    )
    loaded = _load(path)
    assert loaded["z"].shape == (N, N, C_Z)
    assert loaded["z"].dtype == np.float16
    assert loaded["pdistogram"].shape == (N, N, N_BINS)
    assert loaded["distogram_bin_boundaries_angstrom"].shape == (N_BINS - 1,)
    assert loaded["distogram_bin_boundaries_angstrom"].dtype == np.float32
    assert "s" not in loaded


def test_esmfold2_esmc_plus_s(tmp_path):
    path = tmp_path / "esmc.npz"
    s = np.random.randn(N, C_S).astype(np.float32)
    esmc = np.random.randn(N, 2560).astype(np.float32)
    write_trunk_artifact(
        path,
        esmfold2_input=_esm_input(),
        structure_id="prot1",
        structure_seed=0,
        model_weight_uri="local://model",
        s=s,
        esmc=esmc,
    )
    loaded = _load(path)
    assert loaded["s"].shape == (N, C_S)
    assert loaded["esmc"].shape == (N, 2560)
    assert loaded["esmc"].dtype == np.float16
    assert int(loaded["esmc_layer_index"]) == -1
    assert int(loaded["esmc_num_layers_available"]) == 81


def test_esmfold2_requires_s_or_z(tmp_path):
    path = tmp_path / "bad.npz"
    esmc = np.random.randn(N, 2560).astype(np.float32)
    with pytest.raises(ValueError, match="at least one of s / z"):
        write_trunk_artifact(
            path,
            esmfold2_input=_esm_input(),
            structure_id="prot1",
            structure_seed=0,
            model_weight_uri="local://model",
            esmc=esmc,
        )


def test_esmfold2_z_without_distogram_fails(tmp_path):
    path = tmp_path / "bad.npz"
    z = np.random.randn(N, N, C_Z).astype(np.float32)
    with pytest.raises(ValueError, match="pdistogram"):
        write_trunk_artifact(
            path,
            esmfold2_input=_esm_input(),
            structure_id="prot1",
            structure_seed=0,
            model_weight_uri="local://model",
            z=z,
        )


# --------------------------------------------------------------------------- #
