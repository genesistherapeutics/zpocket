"""Run trained pocket heads over a frozen-trunk pair rep ``z`` to get per-residue pocket probs.

Read a features NPZ (key ``z``, shape ``(L, L, C)``), score one or more head checkpoints,
mean-ensemble their logits, and squash with a sigmoid to per-residue pocket probabilities
``(L,)``.
"""

from __future__ import annotations

import numpy as np
import torch

from zpocket.heads.pairformer import ZPairformerHead, load_pairformer_head


def _z_logits(z: np.ndarray, head: ZPairformerHead) -> np.ndarray:
    """Per-residue logits ``(L,)`` for one head over a pair rep ``z`` (CPU, no grad, fp32)."""
    zt = torch.from_numpy(np.ascontiguousarray(z)).float()
    with torch.no_grad():
        return head(zt).float().cpu().numpy().reshape(-1)


def predict_pockets(z: np.ndarray, head: ZPairformerHead) -> np.ndarray:
    """Per-residue pocket probabilities ``(L,)`` from one head over pair rep ``z`` (L, L, C)."""
    return _sigmoid(_z_logits(z, head))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float64)


def _mean_ensemble(per_head_logits: list[np.ndarray]) -> np.ndarray:
    """Mean of per-head logits (a 1-element list is returned unchanged)."""
    if len(per_head_logits) == 1:
        return per_head_logits[0]
    return np.mean(np.stack(per_head_logits, axis=0), axis=0)


def _read_features_npz(features_npz: str) -> dict:
    """Read a zpocket features NPZ: the pair rep ``z`` (L, L, C) plus optional sequence metadata.

    Returns ``{"z", "sequence_sha256", "sequence"}`` (the sha256 / sequence are ``None`` if absent).
    ESMFold2 NPZs key the digest as ``sequence_sha256``.
    """
    with np.load(features_npz, allow_pickle=True) as npz:
        if "z" not in npz.files:
            raise KeyError(f"{features_npz}: no 'z' array (keys: {sorted(npz.files)})")
        z = np.asarray(npz["z"])
        sha = None
        for key in ("sequence_sha256", "protein_sequence_sha256"):
            if key in npz.files:
                sha = _scalar(npz[key])
                break
        sequence = _scalar(npz["sequence"]) if "sequence" in npz.files else None
    return {"z": z, "sequence_sha256": sha, "sequence": sequence}


def _scalar(arr) -> str | None:
    """Coerce a 0-d / 1-element NPZ string array to a plain ``str`` (or ``None``)."""
    val = np.asarray(arr)
    if val.ndim == 0:
        val = val.item()
    else:
        flat = val.reshape(-1)
        val = flat[0] if flat.size else None
    if val is None:
        return None
    return val.decode() if isinstance(val, bytes) else str(val)


def predict_from_npz(features_npz: str, heads: list[ZPairformerHead]) -> dict:
    """Ensemble ``heads`` over the ``z`` in ``features_npz`` → per-residue pocket probs.

    Mirrors ``iter_fullz_head_logits``: each head scores the shared ``z`` once, the per-head logits
    are mean-ensembled, then sigmoid'd. Returns ``{"probs" (L,), "sequence_sha256", "sequence"}``.
    """
    feats = _read_features_npz(features_npz)
    per_head = [_z_logits(feats["z"], h) for h in heads]
    probs = _sigmoid(_mean_ensemble(per_head))
    return {
        "probs": probs,
        "sequence_sha256": feats["sequence_sha256"],
        "sequence": feats["sequence"],
    }


def load_heads(ckpt_paths: list[str]) -> list[ZPairformerHead]:
    """Load an ensemble of pairformer heads from checkpoint paths (all must share ``pair_dim``)."""
    heads, metas = [], []
    for path in ckpt_paths:
        head, meta = load_pairformer_head(path)
        heads.append(head)
        metas.append(meta)
    pair_dims = {m["pair_dim"] for m in metas}
    if len(pair_dims) > 1:
        raise ValueError(f"ensemble heads disagree on pair_dim: {sorted(pair_dims)}")
    return heads
