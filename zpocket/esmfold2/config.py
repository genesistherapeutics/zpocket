"""ESMFold2 folding configuration for zpocket.

`ESMFold2Config` **defaults to the published recipe** (`PAPER_SAMPLING`), so a plain
`zpocket fold` / `zpocket run` reproduces the features the released head was trained to
read. Every knob can be overridden individually from the CLI.

This is deliberately not `esm`'s own set of defaults, which differ on sampling and,
since esm 3.4, on MSA and atom-attention semantics. `z` is recomputed every trunk loop,
so those differences produce a genuinely different feature tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

# The trunk + sampling recipe the published zPocket release records as the one its features and
# folds were produced with (genesisml/zPocket config.json, `feature_provenance`). These are NOT
# esm's own fold() defaults, which have also moved between esm versions -- so pin them rather
# than inherit them. `z` is the head's only input and is recomputed every trunk loop, so
# `num_loops` and `lm_dropout` change the prediction itself, not just the output structure;
# the diffusion knobs act after `z` and change only the predicted coordinates (which still
# moves the centroid metrics, since those read the fold).
PAPER_SAMPLING = {
    "num_loops": 10,
    "num_sampling_steps": 100,
    "noise_scale": 1.003,
    "step_scale": 1.5,
    "max_inference_sigma": 256.0,
    "num_diffusion_samples": 3,
    "seed": 0,
    "lm_dropout": 0.25,
}

PUBLISHED_MSA_MAX_DEPTH = 16384
PUBLISHED_MSA_COLUMN_MASK_RATE = 0.0


@dataclass
class ESMFold2Config:
    """Knobs for a single-chain ESMFold2 fold + optional trunk-feature capture.

    Basic knobs (always forwarded):
        seed: Base diffusion seed.
        num_diffusion_samples: Number of poses generated per input.
        use_msa: When False, the model runs single-sequence (any attached MSA is
            dropped). Default True.
        published_checkpoint_compatibility: Reproduce the historical ESMFold2
            runtime semantics used to train the released zPocket head. This zeroes
            MSA deletion features, consumes the complete unmasked MSA, and uses full
            atom attention. Default True.

    Sampling knobs default to the published recipe (`PAPER_SAMPLING`). Only knobs the
    installed ``esm`` actually accepts are forwarded, so this config works across esm
    versions whose ``fold()`` signatures differ.

    LM dropout:
        lm_dropout: ESMFold2 keeps its language-model dropout active at inference, with a
            fresh mask per trunk loop, so ``z`` is stochastic even at a fixed seed. The
            published features used ``0.25``. **Note that ``0`` does not disable dropout**
            — esm treats a falsy value as "leave the checkpoint config untouched", which
            still applies the checkpoint's own rate.

    MSA subsampling:
        msa_max_depth / msa_column_mask_rate: Explicit overrides. When either is
            ``None``, published compatibility resolves it to 16,384 / 0.0; modern
            esm semantics leave it unset and therefore use esm's 1,024 / 0.1 defaults.

    Feature-save flags:
        save_trunk_single: Persist the per-residue single representation ``s``
            (``inputs_embedder`` output).
        save_trunk_embeddings: Persist the pair representation ``z`` +
            distogram logits (``parcae_coda`` / ``distogram_head`` output).
        save_esmc_embeddings: Persist the ESMC-6B last-layer per-residue embedding.
            ESMFold2-only; requires at least one of s / z to also be saved.

    ESMC precision:
        esmc_precision: Precision for the ESMC-6B backbone. Default ``"bf16"``, which is
            what the published features were built with. If you observe NaNs or gesvd SVD
            non-convergence, switch to ``"fp32"``.
    """

    # Basic knobs (always forwarded to builder.fold / input build).
    seed: int = 0
    num_diffusion_samples: int = 3
    use_msa: bool = True
    published_checkpoint_compatibility: bool = True

    # Sampling recipe. Defaults are the published ones (PAPER_SAMPLING).
    num_loops: int = 10
    num_sampling_steps: int = 100
    noise_scale: float = 1.003
    step_scale: float = 1.5
    max_inference_sigma: float = 256.0

    # Language-model dropout. Active at inference by design; 0.25 is the published value.
    lm_dropout: float | None = 0.25

    # MSA overrides. None selects the active semantic mode's default.
    msa_max_depth: int | None = None
    msa_column_mask_rate: float | None = None

    # Frozen-trunk feature-save flags.
    save_trunk_single: bool = False
    save_trunk_embeddings: bool = False
    save_esmc_embeddings: bool = False

    # ESMC-6B backbone load precision ("bf16" validated; "fp32" if NaNs appear).
    esmc_precision: str = "bf16"

    @property
    def save_trunk(self) -> bool:
        return self.save_trunk_single or self.save_trunk_embeddings or self.save_esmc_embeddings

    @property
    def effective_msa_max_depth(self) -> int | None:
        if self.msa_max_depth is not None:
            return self.msa_max_depth
        if self.published_checkpoint_compatibility:
            return PUBLISHED_MSA_MAX_DEPTH
        return None

    @property
    def effective_msa_column_mask_rate(self) -> float | None:
        if self.msa_column_mask_rate is not None:
            return self.msa_column_mask_rate
        if self.published_checkpoint_compatibility:
            return PUBLISHED_MSA_COLUMN_MASK_RATE
        return None
