"""Compatibility helpers for the ESMFold2 stack used to train the published head."""

from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import esm.models.esmfold2.layers as esmfold2_layers
import torch
import torch.nn.functional as F
from torch import Tensor


def _published_atom_attention_forward(self: Any, x: Tensor, attention_params: tuple) -> Tensor:
    """Run the full-attention fallback used by the published training stack."""
    batch_size, num_atoms = x.shape[:2]
    cos, sin = attention_params[0], attention_params[1]

    x_input = x
    qkv = self.Wqkv(x)
    qkv = qkv.view(batch_size, num_atoms, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
    q, k, v = qkv.unbind(0)
    q = esmfold2_layers.qk_norm(q)
    k = esmfold2_layers.qk_norm(k)

    q = esmfold2_layers.apply_rotary_emb_3d(q, cos, sin)
    k = esmfold2_layers.apply_rotary_emb_3d(k, cos, sin)

    input_dtype = q.dtype
    if q.dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    # The historical transformers fork used full attention without a padding mask
    # whenever flash-attn was unavailable. The published feature jobs took this path.
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    attention = F.softmax(attention, dim=-1)
    out = torch.matmul(attention, v).transpose(1, 2)

    out = out.to(input_dtype).reshape(batch_size, num_atoms, -1)
    out = out * torch.sigmoid(self.gate_proj(x_input))
    return self.out_proj(out)


@contextmanager
def published_atom_attention(model: Any) -> Iterator[None]:
    """Temporarily apply the published full-attention behavior to one model.

    Patching individual module instances keeps unrelated ESMFold2 models in the
    process on their installed ``esm`` behavior and restores every module even if
    folding raises.
    """
    patched: list[tuple[Any, bool, Any]] = []
    for module in model.modules():
        if not isinstance(module, esmfold2_layers.SWA3DRoPEAttention):
            continue
        had_instance_forward = "forward" in module.__dict__
        original_forward = module.__dict__.get("forward")
        module.forward = types.MethodType(_published_atom_attention_forward, module)
        patched.append((module, had_instance_forward, original_forward))

    try:
        yield
    finally:
        for module, had_instance_forward, original_forward in patched:
            if had_instance_forward:
                module.forward = original_forward
            else:
                del module.forward
