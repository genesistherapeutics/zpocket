"""GPU smoke test: fold one tiny sequence if weights + CUDA are present.

Skipped unless CUDA is available AND the weight-location env vars point at real weights. These are faithful-but-unrun on CI without a GPU; run manually with:

  ESMFOLD2_MODEL_DIR=... ESMFOLD2_ESMC_DIR=... ESMFOLD2_CCD=/path/ccd.pkl \\
    pytest tests/test_smoke_gpu.py -q -k esmfold2
"""

from __future__ import annotations

import os

import pytest

TINY_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


def _no_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return True
    return not torch.cuda.is_available()


_ESM_ENV = ("ESMFOLD2_MODEL_DIR", "ESMFOLD2_ESMC_DIR", "ESMFOLD2_CCD")


@pytest.mark.skipif(_no_cuda(), reason="no CUDA")
@pytest.mark.skipif(
    not all(os.environ.get(k) for k in _ESM_ENV),
    reason="ESMFOLD2_MODEL_DIR / ESMFOLD2_ESMC_DIR / ESMFOLD2_CCD not set",
)
def test_esmfold2_smoke(tmp_path):
    from zpocket.esmfold2.config import ESMFold2Config
    from zpocket.esmfold2.fold import fold_protein
    from zpocket.esmfold2.loader import load_esmfold2

    model_dir = os.environ["ESMFOLD2_MODEL_DIR"]
    model, builder = load_esmfold2(
        model_dir, os.environ["ESMFOLD2_ESMC_DIR"], os.environ["ESMFOLD2_CCD"]
    )
    cfg = ESMFold2Config(num_diffusion_samples=1, save_trunk_single=True)
    out = fold_protein(
        model,
        builder,
        sequence=TINY_SEQ,
        structure_id="smoke",
        out_dir=str(tmp_path),
        cfg=cfg,
        model_weight_uri=model_dir,
        a3m_path=None,
    )
    assert out["trunk_npz"] and os.path.exists(out["trunk_npz"])
    assert out["cif_paths"]
