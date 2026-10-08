"""Read-only forward hooks that capture ESMFold2 trunk tensors during fold().

The hook module paths (``inputs_embedder`` -> s, ``parcae_coda`` -> z, ``distogram_head``
-> pdistogram, ``language_model`` input -> esmc) are the capture points the published
features were taken at, so features captured here match the published ones. ``parcae_coda``
is the one the pocket heads read; see the ``genesisml/zPocket`` model card
(``feature_provenance.capture_point``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def capture_trunk_predictions(
    model: Any, *, capture_single: bool, capture_pair: bool, capture_esmc: bool
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture selected trunk tensors without changing the forward pass.

    When ``capture_single`` the per-residue single representation ``s`` (the
    ``inputs_embedder`` output) is captured; when ``capture_pair`` the final pair
    tensor ``z`` (``parcae_coda`` output) and the raw distogram logits
    (``distogram_head`` output) are captured; when ``capture_esmc`` the ESMC-6B
    per-residue hidden-state stack that FEEDS ``language_model`` (its ``inputs[0]``,
    shape ``[B, L, 81, 2560]``) is captured. Each hook is registered only for the
    representations requested, so the s-only path pays nothing for the pair hooks.
    """
    captured: dict[str, torch.Tensor] = {}

    def _capture_s(_module: Any, _inputs: Any, output: Any) -> None:
        captured["s"] = output[0].detach()

    def _capture_z(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured["z"] = output.detach()

    def _capture_pdistogram(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured["pdistogram"] = output.detach()

    def _capture_esmc(_module: Any, inputs: Any, _output: Any) -> None:
        captured["esmc"] = inputs[0].detach()

    handles = []
    if capture_single:
        handles.append(model.inputs_embedder.register_forward_hook(_capture_s))
    if capture_pair:
        handles.append(model.parcae_coda.register_forward_hook(_capture_z))
        handles.append(model.distogram_head.register_forward_hook(_capture_pdistogram))
    if capture_esmc:
        handles.append(model.language_model.register_forward_hook(_capture_esmc))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def distogram_boundaries(model: Any, num_bins: int) -> torch.Tensor:
    """Return the distance boundaries associated with ESMFold2 distogram logits."""
    confidence_cfg = model.config.confidence_head
    return torch.linspace(
        float(confidence_cfg.min_dist),
        float(confidence_cfg.max_dist),
        num_bins - 1,
        dtype=torch.float32,
    )
