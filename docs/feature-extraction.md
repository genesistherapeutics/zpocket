# Feature extraction

Saving frozen-trunk features for training new heads, and scoring saved features on CPU.
For the ordinary path — sequence in, pocket probabilities out — see `zpocket run` in the
[README](../README.md).

Use `fold` to save frozen-trunk features for training new heads or scoring saved
features:

```bash
zpocket fold --model esmfold2 \
    --fasta proteins.fasta --out out/ --no-msa \
    --save s,z,esmc \
    --model-dir weights/esmfold2 --esmc-dir weights/esmc-6b --ccd weights/esmfold2/ccd.pkl
```

Both `fold` and `run` use the published recipe below. These differ from `esm.fold()`
defaults; each command logs the recipe it uses. For benchmark features, supply the
published MSAs rather than `--no-msa`.

| Flag | Default | Effect |
|---|---|---|
| `--num-loops` | 10 (11 total iterations) | Changes `z` |
| `--lm-dropout` | 0.25 | Active at inference, fresh mask per loop; changes `z` |
| `--num-sampling-steps` | 100 | Structure only |
| `--noise-scale` / `--step-scale` | 1.003 / 1.5 | Structure only |
| `--max-inference-sigma` | 256 | Structure only |
| `--samples` / `--seed` | 3 / 0 | Diffusion samples per input / base seed |
| `--msa-max-depth` / `--msa-column-mask-rate` | 16,384 / 0 | Full unmasked MSA used in training |
| `--modern-esm-semantics` | Off | Opt out: preserve deletion counts, use esm MSA defaults and sliding-window atom attention |

**`--lm-dropout 0` does not disable dropout:** `esm` treats zero as leaving the
checkpoint's configured rate unchanged.

`--save` accepts `s`, `z`, `esmc`, and `distogram`; saving `z` also saves `pdistogram`.

| Feature key | Shape | Source |
|---|---|---|
| `s` | `(N, C_s)` | `inputs_embedder` output |
| `z` | `(N, N, C_z)` | `parcae_coda` output; the pocket head's only input |
| `pdistogram` (+ bin boundaries/centres) | `(N, N, B)` | `distogram_head` output |
| `esmc` | `(N, 2560)` | Last-layer ESMC-6B embedding at the ESMFold2 `language_model` input |

The NPZ stores residue metadata and provenance, including sequence and SHA256,
structure ID, seed, model name and weight identity, `tensor_dtype="float16"`, and
schema version. Request at least one of `s` or `z`; `esmc` cannot be saved alone.

Score saved features on CPU with `predict`:

```bash
zpocket predict --head weights/zpocket/model.safetensors \
    --features out/prot1_trunk_predictions.npz --out preds/

# Optional: mean-ensemble logits from multiple heads before the sigmoid.
zpocket predict --head head_a.pt,head_b.pt --features out/*.npz --out preds/
```

Outputs are `<npz basename>_pocket_probs.npz` (`probs`, shape `(L,)`) and a matching
TSV with 1-based `resnum` and `prob`. Heads load from either `model.safetensors` plus
`config.json` or `.pt` training checkpoints, with strict state-dict loading.
