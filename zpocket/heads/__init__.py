"""Pocket-prediction heads: plain-torch readouts over a frozen-trunk pair rep ``z``.

Plain-torch ports of the trained pocket heads, with no cuEquivariance dependency. They
load the published checkpoints and map ``z`` (L, L, C) to a per-residue pocket logit (L,).
The flagship head is :class:`ZPairformerHead`.
"""

from zpocket.heads.pairformer import (
    PairformerLayerNoSeq,
    Transition,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    ZPairformerHead,
    load_pairformer_head,
)

__all__ = [
    "PairformerLayerNoSeq",
    "Transition",
    "TriangleMultiplicationIncoming",
    "TriangleMultiplicationOutgoing",
    "ZPairformerHead",
    "load_pairformer_head",
]
