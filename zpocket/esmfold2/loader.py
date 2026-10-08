"""Load the ESMFold2 decoder (+ the ESMC-6B backbone and CCD) from local dirs or the Hub.

Everything comes from the ``esm`` package, which carries ESMFold2 natively as of
``esm>=3.4``: ``pip install esm`` is the whole dependency story. zpocket does not import
``transformers`` at all — ``esm`` depends on a stock release of it, and nothing here needs
the private fork earlier versions of this package were pinned to.

Two checkpoint layouts exist on ``biohub/ESMFold2`` and both load:

``PINNED_REVISION``
    Decoder only (~235 M parameters); the ESMC-6B backbone is a separate download, attached
    with ``model.load_esmc``. **This is the revision the published zPocket features were
    produced with** — the ``genesisml/zPocket`` model card records its weight digest — so it
    is the default here.
``main``
    Bundles ESMC-6B into the checkpoint (~6.58 B parameters), so no separate backbone
    download is needed. Pass ``revision="main"`` and omit ``esmc_dir``.

.. note::
   The model card lists the trunk's ``model.safetensors`` and ``config.json`` sha256 digests
   the other way round: its ``model.safetensors`` value is in fact ``config.json``'s digest
   on `PINNED_REVISION`, and vice versa. The weights themselves are that revision's.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

# The `biohub/ESMFold2` revision whose weights the published zPocket features were produced
# with. Only consulted when `model_dir` is a repo id rather than a local directory; pin a
# local download yourself.
PINNED_REVISION = "e1e189d0f5fb70c2693da2332eca4443c0ccccd6"


def load_esmfold2(
    model_dir: str,
    esmc_dir: str | None = None,
    ccd_path: str | None = None,
    *,
    esmc_precision: str = "bf16",
    revision: str | None = None,
    device: str = "cuda",
) -> tuple[Any, ESMFold2InputBuilder]:
    """Load the model, attach the ESMC-6B backbone, and build its input builder.

    Args:
        model_dir: Local ESMFold2 checkpoint directory, or an HF repo id.
        esmc_dir: Local ESMC-6B directory (or repo id) to attach. ``None`` lets the
            checkpoint supply its own backbone, which only works on a revision that bundles
            one (see the module docstring) — on a decoder-only checkpoint ``esm`` resolves
            the backbone named in its config, which may hit the network.
        ccd_path: Path to ``ccd.pkl``. Its parent directory is used as the builder's
            ``ccd_cache`` (which expects ``<dir>/ccd.pkl``), so the file MUST be named
            ``ccd.pkl``. ``None`` lets ``esm`` resolve the dictionary itself.
        esmc_precision: ESMC-6B precision. ``"bf16"`` is the published setting. Switch to
            ``"fp32"`` if you see NaNs or gesvd SVD non-convergence.
        revision: Hub revision when `model_dir` is a repo id. ``None`` uses
            `PINNED_REVISION`. Ignored for a local directory.
        device: Device to place the model on.

    Returns:
        ``(model, builder)`` — the model on `device` in eval mode.
    """
    from_pretrained_kwargs: dict[str, Any] = {}
    if not os.path.isdir(model_dir):
        from_pretrained_kwargs["revision"] = revision or PINNED_REVISION

    # load_esmc=False defers the backbone to the explicit by-path load below; when no path was
    # given, let the checkpoint bring its own.
    model = EsmFold2Model.from_pretrained(
        model_dir,
        load_esmc=esmc_dir is None,
        esmc_precision=esmc_precision,
        device=device,
        **from_pretrained_kwargs,
    ).eval()
    if esmc_dir is not None:
        model.load_esmc(esmc_dir, precision=esmc_precision)

    ccd_cache = None
    if ccd_path is not None:
        ccd_path = os.path.abspath(ccd_path)
        # ESMFold2InputBuilder(ccd_cache=dir) loads `<dir>/ccd.pkl`; the env var lets esm find
        # the same dictionary without reaching for HuggingFace.
        os.environ["ESMCFOLD_CCD_PATH"] = ccd_path
        ccd_cache = Path(os.path.dirname(ccd_path))

    return model, ESMFold2InputBuilder(ccd_cache=ccd_cache)
