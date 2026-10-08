"""Fold one protein with ESMFold2 and write its trunk-feature NPZ + CIF(s).

One fold, one NPZ, one CIF per diffusion sample. There is no per-seed best-of-N
selection ranking, no ligand handling and no cache or OOM-recovery logic. The
s/z/esmc/distogram extraction (fp16 CPU casts, last-layer esmc slice) matches how the
published features were produced.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch

from zpocket.esmfold2.capture import capture_trunk_predictions, distogram_boundaries
from zpocket.esmfold2.config import ESMFold2Config
from zpocket.esmfold2.input import build_esmfold2_input_dict, build_spi
from zpocket.esmfold2.trunk_artifacts import (
    single_chain_protein_metadata,
    write_trunk_artifact,
)
from zpocket.exceptions import ZPocketError

logger = logging.getLogger(__name__)


def _fold_kwargs(builder: Any, cfg: ESMFold2Config, structure_id: str) -> dict[str, Any]:
    """The recipe to pass to ``builder.fold``, filtered to the knobs it actually accepts.

    `ESMFold2Config` defaults to the published recipe, so every sampling knob is forwarded
    explicitly rather than inherited from ``esm``'s own ``fold()`` defaults — those differ
    between esm versions and are not the published ones.

    Knobs the installed ``esm`` does not expose are dropped instead of raising, so one config
    works across esm releases whose signatures differ. A dropped knob is not silently ignored:
    it is reported, because an un-forwarded `num_loops` or `lm_dropout` changes `z`.

    ``max_inference_sigma`` is passed through WITHOUT an int() cast: the sampler casts float()
    internally, so an int() cast here would silently truncate a fractional override.
    """
    wanted: dict[str, Any] = {
        "num_diffusion_samples": cfg.num_diffusion_samples,
        "seed": cfg.seed,
        "complex_id": str(structure_id),
        "num_loops": cfg.num_loops,
        "num_sampling_steps": cfg.num_sampling_steps,
        "noise_scale": cfg.noise_scale,
        "step_scale": cfg.step_scale,
        "max_inference_sigma": cfg.max_inference_sigma,
    }
    # None means "defer to the checkpoint's own config", which is the published behaviour, so
    # these two are only forwarded when the user explicitly set them.
    for name in ("lm_dropout", "msa_max_depth", "msa_column_mask_rate"):
        value = getattr(cfg, name)
        if value is not None:
            wanted[name] = value

    accepted = _fold_parameters(builder)
    if accepted is None:
        return wanted
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    dropped = sorted(set(wanted) - set(kwargs))
    if dropped:
        logger.warning(
            "this esm's ESMFold2InputBuilder.fold() does not accept %s; those settings are "
            "NOT applied, so the features will differ from the published ones",
            ", ".join(dropped),
        )
    return kwargs


def _fold_parameters(builder: Any) -> set[str] | None:
    """Parameter names ``builder.fold`` accepts, or None if it takes ``**kwargs``."""
    try:
        signature = inspect.signature(builder.fold)
    except (TypeError, ValueError):  # a C-implemented or otherwise opaque callable
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return None
    return set(signature.parameters)


def fold_protein(
    model: Any,
    builder: Any,
    *,
    sequence: str,
    structure_id: str,
    out_dir: str,
    cfg: ESMFold2Config,
    model_weight_uri: str,
    a3m_path: str | None = None,
    chain_id: str = "A",
) -> dict[str, list[str] | str | None]:
    """Fold a single protein chain and persist its outputs.

    Returns a dict with ``trunk_npz`` (path or None) and ``cif_paths`` (list).
    """
    esmfold2_input = build_esmfold2_input_dict(sequence, chain_id=chain_id)
    if cfg.save_trunk:
        # Validate the supported single-chain protein mapping before paying for a fold.
        single_chain_protein_metadata(esmfold2_input)

    spi = build_spi(esmfold2_input, a3m_path=a3m_path if cfg.use_msa else None)

    with ExitStack() as stack:
        captured = (
            stack.enter_context(
                capture_trunk_predictions(
                    model,
                    capture_single=cfg.save_trunk_single,
                    capture_pair=cfg.save_trunk_embeddings,
                    capture_esmc=cfg.save_esmc_embeddings,
                )
            )
            if cfg.save_trunk
            else {}
        )
        fold_out = builder.fold(model, spi, **_fold_kwargs(builder, cfg, structure_id))

    trunk_npz: str | None = None
    if cfg.save_trunk:
        # The fold and confidence heads have finished. Cast detached copies directly
        # to fp16 on CPU. Each representation is captured only when its flag is set,
        # so missing means the hook never fired (a real failure).
        missing: list[str] = []
        s_numpy = None
        if cfg.save_trunk_single:
            s = captured.get("s")
            if s is None:
                missing.append("s")
            else:
                s_numpy = s.to(device="cpu", dtype=torch.float16).numpy()

        z_numpy = None
        pdistogram_numpy = None
        boundaries_numpy = None
        if cfg.save_trunk_embeddings:
            z = captured.get("z")
            pdistogram = captured.get("pdistogram")
            if z is None or pdistogram is None:
                missing.extend(sorted({"z", "pdistogram"} - set(captured)))
            else:
                z_numpy = z[0].to(device="cpu", dtype=torch.float16).numpy()
                pdistogram_numpy = pdistogram[0].to(device="cpu", dtype=torch.float16).numpy()
                boundaries_numpy = distogram_boundaries(model, pdistogram_numpy.shape[-1]).numpy()

        esmc_numpy = None
        if cfg.save_esmc_embeddings:
            esmc = captured.get("esmc")
            if esmc is None:
                missing.append("esmc")
            else:
                # esmc is the [B, L, 81, 2560] hidden-state stack feeding
                # language_model; keep the LAST ESMC layer only -> (L, 2560).
                esmc_numpy = esmc[0][:, -1, :].to(device="cpu", dtype=torch.float16).numpy()

        if missing:
            raise ZPocketError(
                f"ESMFold2 did not expose captured tensors {missing} for {structure_id} "
                f"seed {cfg.seed}"
            )

        trunk_npz = os.path.join(out_dir, f"{structure_id}_trunk_predictions.npz")
        write_trunk_artifact(
            trunk_npz,
            esmfold2_input=esmfold2_input,
            structure_id=str(structure_id),
            structure_seed=cfg.seed,
            model_weight_uri=model_weight_uri,
            s=s_numpy,
            z=z_numpy,
            pdistogram=pdistogram_numpy,
            distogram_bin_boundaries_angstrom=boundaries_numpy,
            esmc=esmc_numpy,
        )

    samples = fold_out if isinstance(fold_out, list) else [fold_out]
    cif_paths: list[str] = []
    for sample_idx, sample in enumerate(samples):
        cif_path = os.path.join(out_dir, f"{structure_id}_sample{sample_idx}.cif")
        with open(cif_path, "w") as fh:
            fh.write(sample.complex.to_mmcif())
        cif_paths.append(cif_path)

    return {"trunk_npz": trunk_npz, "cif_paths": cif_paths}


def fold_capture_z(
    model: Any,
    builder: Any,
    *,
    sequence: str,
    structure_id: str,
    out_dir: str,
    cfg: ESMFold2Config,
    a3m_path: str | None = None,
    chain_id: str = "A",
    write_cif: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Fold one chain and hand back the pair rep ``z`` in memory, never writing it to disk.

    `fold_protein` persists every requested representation as an NPZ, which is the right contract
    when the features are the product. When the *scores* are the product it is the wrong one: ``z``
    is quadratic in length — ~66 MB at L=360 and ~737 MB at L=1,200 in float16 — so a whole split's
    NPZs cost well over a hundred gigabytes to produce score vectors of a few kilobytes each. This
    folds, captures ``z``, and leaves the caller to score it and drop it.

    The fold itself is still written when `write_cif`, because the centroid metrics need it and it
    is small.

    Args:
        model: Loaded ESMFold2 model.
        builder: Its `ESMFold2InputBuilder`.
        sequence: One-letter chain sequence.
        structure_id: Name used for the output CIFs.
        out_dir: Directory for the CIFs.
        cfg: Fold configuration. Its feature-save flags are ignored — the pair hooks are always
            the ones registered here.
        a3m_path: Precomputed alignment, or None for single-sequence folding.
        chain_id: Chain label for the built input.
        write_cif: Persist the predicted structure(s). False skips them, for a scores-only run.

    Returns:
        ``(z, cif_paths)`` — ``z`` of shape ``(L, L, C)`` as float32, and the CIFs written.

    Raises:
        ZPocketError: If the pair hook never fired, which means the capture module path no longer
            matches the model and the features would silently be absent.
    """
    esmfold2_input = build_esmfold2_input_dict(sequence, chain_id=chain_id)
    single_chain_protein_metadata(esmfold2_input)
    spi = build_spi(esmfold2_input, a3m_path=a3m_path if cfg.use_msa else None)

    with capture_trunk_predictions(
        model, capture_single=False, capture_pair=True, capture_esmc=False
    ) as captured:
        fold_out = builder.fold(model, spi, **_fold_kwargs(builder, cfg, structure_id))

    z = captured.get("z")
    if z is None:
        raise ZPocketError(
            f"ESMFold2 did not expose the pair representation for {structure_id}; the "
            f"`parcae_coda` capture hook never fired"
        )
    # float32 on CPU: the head runs in fp32, so casting through fp16 would only lose precision.
    z_numpy = z[0].to(device="cpu", dtype=torch.float32).numpy()

    cif_paths: list[str] = []
    if write_cif:
        samples = fold_out if isinstance(fold_out, list) else [fold_out]
        os.makedirs(out_dir, exist_ok=True)
        for sample_idx, sample in enumerate(samples):
            cif_path = os.path.join(out_dir, f"{structure_id}_sample{sample_idx}.cif")
            with open(cif_path, "w") as fh:
                fh.write(sample.complex.to_mmcif())
            cif_paths.append(cif_path)
    return z_numpy, cif_paths
