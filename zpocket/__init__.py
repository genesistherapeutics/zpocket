"""zpocket: standalone protein folding + frozen-trunk feature extraction.

Wraps ESMFold2 (the ``esm`` package) to fold single-chain proteins and persist
their frozen-trunk representations (single ``s``, pair ``z`` + distogram logits,
and the ESMC-6B last-layer embedding) as compressed float16 NPZ artifacts.

No object-store clients and no ligand code. Model weights load from local
directories (or HuggingFace); MSAs are supplied as precomputed a3m files (one
per sequence) or omitted for single-sequence folding.
"""

from zpocket.exceptions import ZPocketError

__all__ = ["ZPocketError"]
