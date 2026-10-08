"""Plain-torch port of the triangle-Pairformer pocket head (``ZPairformerHead``).

A faithful port of the trained architecture — ``PairformerLayerNoSeq`` with its
triangle-multiplication and transition building blocks — with no cuEquivariance dependency.
The forward path is the plain-einsum one, so it runs identically on CPU or GPU in plain fp32.
The published ``genesisml/zPocket`` model card states the forward pass this implements
step by step (``architecture.forward``).

Parameter names and shapes match the trained checkpoints exactly, so
:func:`load_pairformer_head` performs a ``strict=True`` load with zero missing / unexpected keys.
Every ``LayerNorm`` uses ``eps=1e-5``, the value the weights were trained under.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# The eps the weights were trained under (1e-05). Using a different eps would
# silently perturb every normalized activation even though load_state_dict would still succeed.
_LN_EPS = 1e-5


class TriangleMultiplicationOutgoing(nn.Module):
    """Triangle multiplication (outgoing edges), plain-einsum forward.

    Normalize, gated input projection, mask, split into ``a, b`` and mix along the shared
    ``k`` axis with the outgoing einsum ``bikd,bjkd->bijd``, then gated output projection.
    A plain sum over ``k`` — there is no ``1/L`` scaling.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=_LN_EPS)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim, eps=_LN_EPS)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B, N, N, D); mask: (B, N, N)
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x


class TriangleMultiplicationIncoming(nn.Module):
    """Triangle multiplication (incoming edges), plain-einsum forward.

    Identical to :class:`TriangleMultiplicationOutgoing` except the triangular projection mixes
    along the shared axis with the incoming einsum ``bkid,bkjd->bijd``.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=_LN_EPS)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim, eps=_LN_EPS)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = torch.einsum("bkid,bkjd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x


class Transition(nn.Module):
    """Two-layer SwiGLU MLP: ``fc3(silu(fc1(norm(x))) * fc2(norm(x)))``.

    Norm is applied once; ``fc1`` and ``fc2`` both consume the normed input; ``fc3`` projects the
    gated hidden back to ``out_dim``. All linears are bias-free (matches the checkpoint keys).
    """

    def __init__(self, dim: int = 128, hidden: int = 512, out_dim: int | None = None) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.norm = nn.LayerNorm(dim, eps=_LN_EPS)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class PairformerLayerNoSeq(nn.Module):
    """Pairformer layer that only updates ``z`` (no sequence embeddings).

    Residual add of outgoing triangle-mult, incoming triangle-mult, and the SwiGLU transition
    (transition hidden = ``4 * token_z``). Dropout is a no-op at eval time but is kept as an
    attribute so the module tree (and therefore the state_dict) matches the trained one.
    """

    def __init__(self, token_z: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        z = z + self.dropout(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.transition_z(z)
        return z


class ZPairformerHead(nn.Module):
    """Per-residue pocket head: project ``z`` to ``z_dim``, run ``n_pairformer`` triangle-Pairformer
    blocks over an all-ones pair mask, then a row/col/diag pooled MLP → per-residue logit.

    ``forward(z)`` takes ``z`` of shape ``(L, L, pair_dim)`` and returns logits of shape ``(L,)``.
    """

    def __init__(
        self, pair_dim: int = 128, z_dim: int = 64, n_pairformer: int = 2, hidden: int = 256
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(pair_dim, z_dim)
        self.pairformer = nn.ModuleList(
            PairformerLayerNoSeq(z_dim, 0.1) for _ in range(n_pairformer)
        )
        self.net = nn.Sequential(
            nn.LayerNorm(3 * z_dim, eps=_LN_EPS),
            nn.Linear(3 * z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        L = z.shape[0]
        z = self.proj(z.float()).unsqueeze(0)  # (1, L, L, z_dim)
        pair_mask = z.new_ones(1, L, L)
        for layer in self.pairformer:
            z = layer(z, pair_mask)
        z = z.squeeze(0)  # (L, L, z_dim)
        idx = torch.arange(L, device=z.device)
        row, col, diag = z.mean(1), z.mean(0), z[idx, idx]  # (L, z_dim) each
        return self.net(torch.cat([row, col, diag], -1)).squeeze(-1)


# Head name -> number of stacked Pairformer blocks, as named in the published checkpoints.
_PAIRFORMER_N = {"pairformer1": 1, "pairformer2": 2, "pairformer3": 3}


def _load_from_safetensors(weights_path: str) -> tuple[ZPairformerHead, dict]:
    """Load the published weights-only release: ``model.safetensors`` + ``config.json``.

    The release carries no torch pickle, so the architecture comes from ``config.json`` beside
    the weights — ``architecture.constructor_kwargs`` gives ``pair_dim`` and ``n_blocks``, and the
    top-level ``head`` names the variant. The published export is bit-identical to the
    ``esmfold2/pw1g0`` training checkpoint and loads into this module strictly (52 tensors,
    330,177 parameters), so the format differs but the weights do not.

    Args:
        weights_path: Path to ``model.safetensors``.

    Returns:
        ``(head, meta)``.

    Raises:
        FileNotFoundError: If no ``config.json`` sits beside the weights. The architecture cannot
            be inferred from the tensors alone without guessing, and a wrong guess would load
            silently into the wrong shape of model.
        ValueError: If the config names a head this module does not implement.
    """
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    config_path = Path(weights_path).with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"{weights_path} has no config.json beside it; the published release ships both and "
            f"the architecture is read from the config rather than inferred from the tensors"
        )
    config = json.loads(config_path.read_text())
    head_name = config.get("head")
    if head_name not in _PAIRFORMER_N:
        raise ValueError(
            f"{config_path}: head {head_name!r} is not a pairformer head "
            f"(expected one of {sorted(_PAIRFORMER_N)})"
        )
    kwargs = config.get("architecture", {}).get("constructor_kwargs", {})
    pair_dim = int(kwargs.get("pair_dim", 256))
    # Prefer the config's own block count over the one implied by the head name; they agree on
    # every published release, but the config is the authority on what the weights actually are.
    n_blocks = int(kwargs.get("n_blocks", _PAIRFORMER_N[head_name]))

    head = ZPairformerHead(pair_dim, n_pairformer=n_blocks)
    head.load_state_dict(load_file(weights_path), strict=True)
    head.eval()
    return head, {
        "head": head_name,
        "pair_dim": pair_dim,
        "n_params": sum(p.numel() for p in head.parameters()),
        "source": "safetensors",
        "trunk": config.get("trunk"),
    }


def load_pairformer_head(ckpt_path: str) -> tuple[ZPairformerHead, dict]:
    """Load a trained ``ZPairformerHead`` (CPU, eval mode), from either published format.

    ``model.safetensors`` is the weights-only release format: the architecture is read from the
    ``config.json`` beside it. A ``.pt`` is a training checkpoint — a dict with ``state_dict``,
    ``head`` (e.g. ``"pairformer2"``) and ``pair_dim``. Either way the head is rebuilt and loaded
    with ``strict=True``, so a missing or unexpected key raises rather than leaving part of the
    model at its initialisation.

    Args:
        ckpt_path: Path to ``model.safetensors`` or to a ``.pt`` checkpoint.

    Returns:
        ``(head, meta)`` where ``meta`` carries ``head``, ``pair_dim`` and ``n_params``.

    Raises:
        ValueError: If the named head is not a pairformer variant.
    """
    if ckpt_path.endswith(".safetensors"):
        return _load_from_safetensors(ckpt_path)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_name = ck.get("head") or ck.get("kind")
    if head_name not in _PAIRFORMER_N:
        raise ValueError(
            f"{ckpt_path}: head {head_name!r} is not a pairformer head "
            f"(expected one of {sorted(_PAIRFORMER_N)})"
        )
    pair_dim = int(ck["pair_dim"])
    head = ZPairformerHead(pair_dim, n_pairformer=_PAIRFORMER_N[head_name])
    head.load_state_dict(ck["state_dict"], strict=True)
    head.eval()
    n_params = sum(p.numel() for p in head.parameters())
    meta = {"head": head_name, "pair_dim": pair_dim, "n_params": n_params, "source": "torch"}
    return head, meta
