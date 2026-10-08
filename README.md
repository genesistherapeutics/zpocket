# zpocket

**Ligand-binding pocket prediction on predicted structures.** A protein goes in; a
per-residue pocket probability and a ranked list of pocket centres come out.

zpocket folds a single-chain protein with **ESMFold2**, reads out the frozen trunk's
pair representation `z`, and runs a small trained `pairformer2` head over it. The trunk
does the work; the head is a 330k-parameter read-out. It also scores itself against the
**PocketWatch** benchmark — per-target and per-pocket AUPRC, and per-centroid DCC.

Weights load from local directories (or HuggingFace); MSAs are supplied as precomputed
a3m files, or omitted for single-sequence folding.

- Model weights: [`genesisml/zPocket`](https://huggingface.co/genesisml/zPocket)
- Dataset: [`genesisml/PocketWatch`](https://huggingface.co/datasets/genesisml/PocketWatch)

> **On Protenix.** The paper also reports these heads on a Protenix-v2 trunk, scoring
> slightly higher. Those weights are no longer publicly available, so the Protenix backend
> has been removed from this package and **ESMFold2 is the only supported trunk**.

### Three things, three commands

| You want | Command | Needs |
|---|---|---|
| pockets on your own protein | `zpocket run` | ESMFold2 + ESMC-6B weights, a head, a GPU |
| the benchmark's test metrics | `zpocket eval` | a scores NPZ — **no GPU, no trunk weights** |
| ground truth in your folds' frame | `zpocket centroids` | your folds |

The split matters: `eval` is pure numpy/sklearn/polars over saved score vectors, so the
metrics can be re-read, re-policied and re-checked without re-running a model.

## Install

```bash
pip install -e .                # runtime only
pip install -e ".[dev]"         # + pytest and ruff, to run the tests
# or, with uv:  uv sync --extra dev
```

Python 3.12, and a CUDA GPU to fold. ESMFold2 ships natively in the `esm` package as of
`esm>=3.4`, so `pip install esm` is the whole dependency story — no forks and no private
git remotes. (Earlier revisions of this package loaded ESMFold2 from a fork of
`transformers`; that is no longer needed, and zpocket does not import `transformers` at
all. `esm` depends on a stock release of it.)

Everything except folding — `eval`, `centroids`, `predict`, and the whole test suite —
needs neither a GPU nor the trunk weights.

## Weights setup (you must supply these)

Model weights are **large and their access/licensing is your responsibility**.

```bash
# the trunk. NOTE the revision pin -- see below.
huggingface-cli download biohub/ESMFold2 \
    --revision e1e189d0f5fb70c2693da2332eca4443c0ccccd6 --local-dir weights/esmfold2
huggingface-cli download biohub/ESMC-6B --local-dir weights/esmc-6b

# the head (~4 MB): model.safetensors + config.json
huggingface-cli download genesisml/zPocket --local-dir weights/zpocket
```

`biohub/ESMFold2` exists in two layouts and zpocket loads either:

| revision | contents | how to load |
|---|---|---|
| `e1e189d0…` (**default**) | decoder only, ~235 M params | pass `--esmc-dir` for the ESMC-6B backbone (~24 GB) |
| `main` | ESMC-6B bundled, ~6.58 B params | omit `--esmc-dir` |

> **Why the pin.** The pinned revision is the one whose weight digest the
> `genesisml/zPocket` model card records, i.e. the weights the published features were
> produced with, so it is what a reproduction should use.
> `zpocket.esmfold2.loader.PINNED_REVISION` applies it automatically when you pass a repo
> id rather than a directory. `main` holds the same trunk re-serialized with the backbone
> bundled; it loads fine but is not the digest the release records.
>
> The model card lists that revision's two sha256 digests **transposed** — the value under
> `model.safetensors` is actually `config.json`'s and vice versa. The weights are the right
> ones; only the labels in the config are swapped.

`--esmc-dir` and `--ccd` are both optional: omit them and `esm` resolves the backbone named
in the checkpoint config and the CCD dictionary itself, which may hit the network. Pass them
to stay offline. If you do pass `--ccd`, the file **must be named `ccd.pkl`** (it ships
inside the ESMFold2 repo); its parent directory becomes the input builder's `ccd_cache`.

> **ESMC-6B precision.** `--esmc-precision bf16` is the default and the published setting.
> If you see NaNs or gesvd SVD non-convergence during folding, use `--esmc-precision fp32`.

---

# Pockets on your own protein

One command. It folds, scores, and writes the answer, without ever persisting the
quadratic pair tensor.

```bash
# --msa-dir reads my_msas/<record_id>.a3m for each FASTA record; use --no-msa instead
# for single-sequence folding (noticeably weaker -- the MSA is the trunk's evolutionary signal).
zpocket run \
    --fasta my_protein.fasta \
    --msa-dir my_msas/ \
    --head weights/zpocket/model.safetensors \
    --model-dir weights/esmfold2 \
    --esmc-dir weights/esmc-6b \
    --ccd weights/esmfold2/ccd.pkl \
    --out results/
```

For each record you get:

| file | what it is |
|---|---|
| `results/<id>_pocket_probs.tsv` | `resnum` (1-based) → `prob`. The answer. |
| `results/<id>_pocket_probs.npz` | the same `(L,)` vector, as `probs` |
| `results/<id>_sample0.cif` | the predicted fold, numbered 1..L, pLDDT in the B-factor column |

### Reading the output

A probability per residue is a ranking, not a yes/no. Two ways to use it:

- **Residues.** Sort the TSV and take the top ~5% — the test-set positive rate is 4.86%,
  so that is roughly "as many residues as a real pocket has". Colour them on the CIF
  (`prob` into the B-factor column) and look.
- **Pocket centres**, if you want a point to dock at rather than a residue set:

  ```python
  from zpocket.discrete import cluster, load_fold
  import numpy as np

  probs = np.load("results/my_protein_pocket_probs.npz")["probs"]
  fold = load_fold("results/my_protein_sample0.cif", "my_protein", len(probs))
  # The published decoder: weighted mean shift, tau=0.3, bandwidth 14 A, min 3 residues.
  for c in cluster(
      probs, fold, method="meanshift", score_threshold=0.3, bandwidth=14.0, min_residues=3
  ):
      print(c.rank, c.center, c.descriptors["n_residues"])
  ```

  Centres come out ranked best-first. This decoder emits ~1.1 pockets per protein on the
  benchmark, so expect a short list. Those values are also `cluster`'s defaults — they are
  spelled out above only to show what is being used. `method="linkage"` is the alternative,
  and scores lower.

**Caveats worth knowing.** The head was trained on single-chain monomer folds, so a pocket
that only exists at a chain interface is not something it can place. Low-pLDDT regions give
unreliable pockets — check the B-factors before trusting a site. And an MSA matters: it is
the trunk's main source of evolutionary signal, and `--no-msa` is noticeably weaker.

---

# The PocketWatch benchmark

[`genesisml/PocketWatch`](https://huggingface.co/datasets/genesisml/PocketWatch) is the
temporally-held-out benchmark these heads are measured on: pockets defined from
Q-BioLiP/PDB contacts at 5 Å, clustered into *sites*, split by **first release date** so a
test pocket is one the field had not seen at the cutoff (2024-01-01).

zpocket pulls what it needs from the Hub on demand, or reads a local tree with
`--dataset-root`. The test split is **1,566 proteins / 1,573 scored pockets** at a 4.86%
positive rate.

### Which pockets count is not your choice

`data/site_policy.parquet` decides, and both halves of the benchmark read that one
artifact — which is what makes per-residue and per-centroid numbers comparable pocket for
pocket. Of its 18,797 sites across all three splits, 14,928 are scored and 3,869 masked:
3,318 as interface-only, 495 as pre-cutoff, 49 with no expressible residue, and 7 as
subsumed by an already-public pocket.

A site the policy excludes is **masked, not negative**: it is still a real pocket, so its
residues stay out of every other pocket's background, and a predicted centre landing on one
is dropped from the ranked list rather than charged as a false positive. Getting this wrong
is the single easiest way to produce numbers that cannot be compared with anything.

### Reproducing the test metrics

```bash
# 0. The test-split alignments (9.7 GiB tar -> 1,566 folders).
huggingface-cli download genesisml/PocketWatch msas_test.tar \
    --repo-type dataset --local-dir dl/
tar -xf dl/msas_test.tar -C work/          # --msa-dir is `work/`, the dir extracted INTO:
                                           # the index spells folders as `msas_test/<hash>`

# 1. Fold + score the test split in one streaming pass. z is never written to disk:
#    it is 66 MB at L=360 and 737 MB at L=1,200, so persisting the split would cost >100 GB.
#    The published trunk recipe (num_loops=10, lm_dropout=0.25, ...) is the DEFAULT.
zpocket run --split-source --split test \
    --head weights/zpocket/model.safetensors \
    --model-dir weights/esmfold2 --esmc-dir weights/esmc-6b --ccd weights/esmfold2/ccd.pkl \
    --msa-dir work/ --msa-layout pocketwatch --msa-combine genesis \
    --out folds/ --scores-out scores.npz

# 2. Score. No GPU, no trunk weights -- just the saved score vectors.
#    The release ships its own ESMFold2 folds and centroids in that frame, so point
#    --structures-dir at them and leave --site-centroids unset.
zpocket eval --scores scores.npz --split test --paper-set --multichain mask \
    --structures-dir <release>/structures \
    --out results.json
```

`eval` prints per-target AUPRC / AUROC / R-precision / hit@1, per-pocket AUPRC (split by
single- vs multi-pocket proteins), and per-centroid DCC at top-N and **top-N+2**, and writes
a JSON carrying one row per scored pocket so any stratification can be applied afterwards.

Add `--no-centroid` to skip the localization half — then no folds are needed at all, which
is the fastest way to check a scores file.

**Shard it across GPUs.** `--shard i/n` deals work round-robin over length-sorted records so
shards finish together. Give each shard its own `--scores-out` and a GPU via
`CUDA_VISIBLE_DEVICES`, then merge the NPZs:

```bash
for i in $(seq 1 8); do
  CUDA_VISIBLE_DEVICES=$((i-1)) zpocket run ... --shard $i/8 --scores-out scores_$i.npz &
done
```

> **Memory.** The longest test proteins reach L=1,198, and ESMFold2 with ESMC-6B resident
> does **not** fit those in 48 GB — on 48 GB cards the longest ~75 proteins fail to fold.
> 96 GB is enough for the whole split. Sharding does not help a single long protein;
> it is per-protein peak memory that binds.
>
> Python buffers stdout when redirected to a file, so a shard's `[run] i/n` progress lines
> appear in bursts. Count `folds/*_sample0.cif` for live progress.

### Which basis you are scoring on

Two published numbers exist for this head and they differ by a **scoring choice, not a
model**. Pick deliberately:

| basis | flags | targets | the number it corresponds to |
|---|---|---|---|
| the paper's table | `--paper-set --multichain mask` | 1,275 (1,562 pockets) | per-target AUPRC **0.745** |
| the checkpoint's own logged metric | `--multichain keep` | 1,559 | per-target AUPRC **0.734** |

`--multichain keep` counts interface-only contacts as positives — which is how the head was
trained and how its own metrics were logged. `mask` removes them, the comparison a
single-chain method is owed, and additionally drops the proteins whose only new pockets are
interface-only. `--paper-set` drops four further test proteins the paper excludes
(`A0A655NVH2`, `P03441`, `Q7XJE8`, `Q962Y6` — non-standard residues the VN-EGNN baseline
cannot read). They are valid targets otherwise; keep them unless you are comparing against
the published table. `tests/test_pocketwatch.py` pins both counts.

### Published reference numbers

The ESMFold2 `pairformer2` row of the paper's main table, on 1,562 pockets of 1,275 targets
(`--paper-set --multichain mask`):

| | AUPRC_t | AUROC_t | Rp_t | AUPRC_p | AUROC_p | Rp_p | 1 pocket | >1 |
|---|---|---|---|---|---|---|---|---|
| zPocket `pairformer2` | 0.745 | 0.951 | 0.682 | 0.702 | 0.941 | 0.639 | 0.751 | 0.599 |
| best baseline (VN-EGNN, retrained) | 0.598 | 0.837 | 0.567 | 0.559 | 0.819 | 0.530 | 0.605 | 0.463 |
| random | 0.049 | 0.500 | 0.049 | 0.051 | 0.500 | 0.051 | 0.053 | 0.048 |

And the checkpoints' own logged test metrics (`--multichain keep`, n=1,559):

| trunk | calibration | AUPRC per-target | AUPRC pooled | AUROC |
|---|---|---|---|---|
| ESMFold2 | `pw1g0` | **0.734** | 0.751 | 0.945 |
| ESMFold2 | `pw5g2` | 0.722 | 0.741 | 0.942 |

The published release is the ESMFold2 `pw1g0` head. `pw1g0` is balanced BCE (best raw
AUPRC); `pw5g2` is focal with `pos_weight=5`, which surfaces more candidate residues at
some cost in precision.

For localization, the paper's `zPocket clusters` row on the same 1,562 pockets of 1,275
targets, DCC at 4 Å:

| | top-N | top-N+2 | ceiling | AP | prec. | k̄ |
|---|---|---|---|---|---|---|
| zPocket clusters, ranked by zPocket | 0.515 | 0.524 | 0.524 | 0.570 | 0.599 | 1.1 |

> **Use the published decoder, or this number is not reachable.** That row is a **weighted
> mean shift** at `score_threshold=0.3`, `bandwidth=14`, `min_residues=3` — chosen by grid
> search on validation and frozen before test was read. Those are the defaults throughout
> this package (`zpocket eval`, and `zpocket.discrete.cluster`). Running `--decode linkage`
> instead tops out around 0.487 top-N+2, and no setting
> of its knobs reaches 0.524: a sweep of 48 `linkage` configurations peaked at 0.4987 even
> when the best was picked *on test*. The gap is the clustering algorithm, not its tuning.
> Bandwidth in particular is not a knob to nudge: mean shift over-fragments badly once the
> bandwidth drops below a pocket's residue shell.

### Why your numbers will not match to the last digit

`z` is **stochastic**, by design, even at a fixed seed: the initial pair state is drawn from
a truncated normal, and the checkpoint's per-loop language-model dropout (`p = 0.25`,
`per_loop_lm_dropout = True`) is active at inference and was left at its default. A
different seed, or different hardware, is a different draw of `z`. Two further deviations
are worth knowing:

- **Precision.** The published predictions were made on the CUDA/bfloat16 path (pairformer
  blocks under autocast, LayerNorms in fp32, readout in fp32). `zpocket predict` and
  `zpocket run` compute the head in float32. The model card puts the difference at up to
  0.010 in probability.
- **ESMC kernels.** Without `transformer_engine`, `xformers` and `flash-attn` installed,
  ESMC-6B falls back to pure-PyTorch LayerNorm, attention and RoPE. `esm` warns about this
  at load; it reports the effect as a few ULP after the final LayerNorm.

Treat the published values as the target to land near, not to match bit for bit.

### The coordinate frame, which matters

Every ligand centroid is expressed in the frame of one specific fold, so a centroid table and
a fold set only mean anything together.

**The release ships ESMFold2 folds and centroids in that frame**, so for the benchmark's own
folds nothing has to be rebuilt — point `--structures-dir` at `structures/` and leave
`--site-centroids` unset.

`zpocket centroids` is what you need for **your own** folds — a different seed, a different
`num_loops`, or another model entirely:

```bash
zpocket centroids --structures-dir folds/ --out site_centroids_mine.parquet
zpocket eval --scores scores.npz --split test --paper-set --multichain mask \
    --structures-dir folds/ --site-centroids site_centroids_mine.parquet
```

The benchmark ships `projection/`, the frame-independent form of the same ground truth: each
observation's ligand centroid in its deposited-entry frame, plus that entry's contact-residue
CA coordinates tagged with canonical position. Superposing those CAs onto your fold (Kabsch,
on the contact residues alone, because a pocket is rigid where distant loops are not) and
carrying the centroid across puts the ground truth in your frame.

Run against the fold set the shipped centroids were built from, it reproduces the shipped
table over all **4,466 sites** with identical member counts — max deviation **1.6e-05 Å**,
median 1.2e-06 Å (`zpocket centroids --compare-to-shipped` reports this). That is a float32
coordinate round-trip, and it is what licenses using the rebuild on a fold set the benchmark
does not ship. `tests/test_centroids.py` asserts the same agreement to 1e-5 Å on a sample of
proteins rather than the whole table, which takes minutes.

The dataset also ships `verify_bundle.py`, a standalone independent check on the same chain
through a separate implementation (numpy + polars only). It rebuilds all 4,466 centroids in
about a minute and agrees to 1.4e-05 Å — two implementations landing in the same place.

### Re-ranking other detectors' proposals — planned for a future release

The paper's strongest localization results do not come from zPocket proposing pockets on its
own. They come from letting a **geometric detector propose** candidate centroids and using the
zPocket head to **re-rank** them. Over the 1,562 test pockets of 1,275 targets, DCC top-N+2 at
4 Å (ESMFold2 `pairformer2` head, every detector run on ESMFold2 folds):

| proposals | ranker | top-N | top-N+2 | ceiling | prec. | k̄ |
|---|---|---|---|---|---|---|
| fpocket | fpocket | 0.139 | 0.197 | 0.476 | 0.014 | 41.3 |
| fpocket | **zPocket** | 0.371 | **0.426** | 0.476 | 0.014 | 41.3 |
| Lacuna (SF-seq) | Lacuna (SF-seq) | 0.273 | 0.497 | 0.780 | 0.015 | 64.9 |
| Lacuna (SF-seq) | **zPocket** | 0.405 | **0.608** | 0.780 | 0.015 | 64.9 |
| VN-EGNN (PW) | VN-EGNN (PW) | 0.533 | 0.602 | 0.620 | 0.179 | 4.2 |
| zPocket clusters | zPocket | 0.515 | 0.524 | 0.524 | **0.599** | 1.1 |
| zPocket + VN-EGNN (PW) | zPocket | **0.544** | **0.675** | 0.690 | 0.213 | 4.0 |

Three findings worth carrying over:

- **Re-ranking alone lifts DCC top-N+2 by +0.090 on average** across six detectors, and
  significantly for every published baseline. The only method it failed to improve was VN-EGNN
  retrained on PocketWatch, whose own ranker is already strong.
- **Adding zPocket's cluster proposals to another method's, then re-ranking, lifts it further**
  — +0.108 on average, and significantly for every method including VN-EGNN (PW). The best
  configuration is zPocket + VN-EGNN (PW) at **0.675**.
- **The clustering decoder is precise but short-sighted.** It emits ~1.1 pockets per protein,
  so it cannot reach a protein's second site and its ceiling is 0.524 — but 60% of what it
  emits lands on a real site, against 18% for retrained VN-EGNN and 4% for P2Rank. That
  precision is why it is useful as a *contributor* of proposals even where its own recall caps
  out. The gap between each row's DCC and its ceiling is the part ranking can still win; the
  ceiling itself is fixed by whatever was proposed.

The same protocol was the one that carried over to the fragment-screening set, where re-ranking
fpocket's cavities with zPocket moved DCC top-N+2 from 0.389 to 0.490.

> **Not in this package yet.** `zpocket eval` ships only the `cluster` decoder
> (`zpocket.discrete.decode`), which reads the head's own scores and nothing else. The
> `rescore` and `hybrid` decoders above need fpocket / P2Rank / Lacuna installed and run over
> every fold, plus the candidate-pooling and de-duplication pipeline that feeds them. **We plan
> to add this pipeline in a future release.** Until then, the numbers in this section are the
> paper's, not something this package reproduces.

### The MSA merge

ESMFold2 takes **one** alignment per chain, but PocketWatch ships **four** per protein. The
merge is not a concatenation, and `zpocket.msa.merge_genesis` implements the one the published
checkpoints were trained on — the recipe the model card records under
`feature_provenance.msa.merge_recipe`. Four details in it change the alignment, so they are
worth knowing if you reimplement it:

1. **Order is `uniprot, uniref90, reduced_bfd, mgnify`** — uniprot first, not deepest-first.
2. **UniProt is deduplicated only against itself**, never against the cross-database set, so
   a sequence found in both UniProt and UniRef90 survives both passes.
3. **The two caps are independent**: UniProt gets at most 8,192 and everything else at most
   `16,383 − 8,192 = 8,191` — a *fixed* bound, not "whatever UniProt left over". A protein
   with 100 UniProt hits yields at most 8,292 sequences, not 16,383.
4. **A final first-wins dedup pass** over the assembled list resolves the duplicates (2) let
   through, in favour of the UniProt copy — which still cost a slot in the other block.

`--msa-combine genesis` is the default and the only setting that reproduces the published
numbers. Naming a single database (`--msa-combine uniref90`) is an ablation.

`tests/test_msa.py` carries an independent transcription of the recipe and asserts **byte
equality** against it on randomized inputs, including at both caps.

---

# Feature extraction

`zpocket fold` persists frozen-trunk features instead of scoring them, which is what you
want to train a new head.

```bash
# ESMFold2: save single + pair + distogram + ESMC, single-sequence
zpocket fold --model esmfold2 \
    --fasta proteins.fasta --out out/ --no-msa \
    --save s,z,esmc \
    --model-dir weights/esmfold2 --esmc-dir weights/esmc-6b --ccd weights/esmfold2/ccd.pkl

# common knobs
#   --seed N       diffusion seed
#   --samples N    diffusion samples per input
#   --save         comma list of: s, z, esmc, distogram   (z implies pdistogram)
```

Both `fold` and `run` default to the **published trunk recipe**, so features you save are
the ones the released head was trained to read:

| knob | default | effect |
|---|---|---|
| `--num-loops` | 10 | trunk iterations (11 total). Recomputes `z`, so it changes the prediction |
| `--lm-dropout` | 0.25 | LM dropout, active at inference with a fresh mask per loop. Changes `z` |
| `--num-sampling-steps` | 100 | diffusion steps — acts after `z`, so structure only |
| `--noise-scale` / `--step-scale` | 1.003 / 1.5 | sampler, structure only |
| `--max-inference-sigma` | 256 | sampler, structure only |
| `--samples` / `--seed` | 3 / 0 | diffusion samples per input; base seed |
| `--msa-max-depth` / `--msa-column-mask-rate` | unset | defer to the checkpoint's own `config.msa_encoder`, which is what the published features used |

These are deliberately **not** `esm`'s own `fold()` defaults, which differ on nearly every
entry (`num_loops=20`, `num_sampling_steps=200`, `num_diffusion_samples=1`,
`lm_dropout=0.3`, `msa_max_depth=1024`, `msa_column_mask_rate=0.1`) and have moved between
esm releases. Each command prints the recipe it is using.

> **`--lm-dropout 0` does not disable dropout.** `esm` reads a falsy value as "leave the
> checkpoint config untouched", which still applies the checkpoint's own rate.

For each folded protein it can persist (all optional, via `--save`):

| feature | key(s) | shape | source |
|---|---|---|---|
| single rep | `s` | `(N, C_s)` | `inputs_embedder` output |
| pair rep | `z` | `(N, N, C_z)` | `parcae_coda` output |
| distogram logits | `pdistogram` (+ bin boundaries/centers) | `(N, N, B)` | `distogram_head` output |
| ESMC-6B embedding | `esmc` | `(N, 2560)` | ESMFold2 `language_model` input, **last layer only** |

`z` from `parcae_coda` is the head's only input. The NPZ also embeds per-residue metadata and
provenance: `sequence`, `sequence_sha256`, `structure_id`, seed, `model_name`, model weight
identity, `tensor_dtype="float16"`, and a `schema_version`. At least one of `s` / `z` must be
requested; `esmc` is supplementary and cannot be saved on its own.

Then score saved features with `zpocket predict` (plain torch, CPU, no trunk stack):

```bash
zpocket predict --head weights/zpocket/model.safetensors \
    --features out/prot1_trunk_predictions.npz --out preds/

# ensemble several seed/replica heads (logits are mean-ensembled before the sigmoid)
zpocket predict --head head_a.pt,head_b.pt --features out/*.npz --out preds/
```

For each input NPZ it writes `preds/<npz basename>_pocket_probs.npz` (key `probs`, shape
`(L,)`) and a matching `.tsv` of `resnum` (1-based) → `prob`.

`load_pairformer_head` reads either published format: the release's `model.safetensors` +
`config.json`, or a `.pt` training checkpoint. Either way the head is rebuilt and loaded
with `strict=True`.

## The head

`ZPairformerHead` projects `z` (L, L, 256) to 64 channels, runs two triangle-Pairformer
blocks (AlphaFold-style triangle multiplication), then a row/col/diagonal pooled MLP → one
logit per residue. 330,177 parameters, `pair_dim=256`. It is plain PyTorch — no ESMFold2
stack and no cuEquivariance — so it runs on CPU.

The model card states the forward pass step by step under `architecture.forward`;
`zpocket/heads/pairformer.py` implements exactly that.

## Tests

```bash
pytest tests/ -q            # everything; needs no GPU. ~30 s with the dataset cached.
```

Needs the `dev` extra (see Install).

Nothing here needs a GPU or trunk weights. Tests that check a *property* build their own
arrays and always run; tests that check a *count* against the published benchmark skip when
it is unreachable (no network, cold cache). Set `POCKETWATCH_ROOT` for a local dataset tree
and `ZPOCKET_HEADS_DIR` for local `.pt` head checkpoints.

| file | what it pins |
|---|---|
| `test_eval_metrics.py` | metric properties: a perfect ranker is 1.0, a random one is the positive fraction, and **a masked residue cannot move any number** (corrupt one and every metric is unchanged) |
| `test_discrete.py` | Kabsch recovers a known transform and never returns a reflection; linkage chains where mean-shift does not; top-N+2 ≥ top-N; a masked site is neither credited nor charged; AP charges for over-emitting |
| `test_pocketwatch.py` | the counts, against the published numbers: 1,566 test proteins, 4.86% positives, 1,573 scored sites, 1,559 scoreable at `multichain=keep`, and **1,562 pockets over 1,275 targets** for the paper set |
| `test_centroids.py` | the centroid rebuild reproduces the shipped table to 1e-5 Å — what licenses scoring localization on folds the benchmark does not ship |
| `test_msa.py` | the merge, byte-equal against an independent transcription of the published recipe |
| `test_fold_recipe.py` | the published recipe actually reaches `fold()` — and a knob the installed `esm` cannot accept is reported, not silently dropped |
| `test_pairformer_head.py` | the release strict-loads with zero missing/unexpected keys, 52 tensors, `n_params == 330177` |
| `test_writers.py` | both trunk NPZ writers round-trip (shapes, dtypes, sha256, schema versions) |
| `test_smoke_gpu.py` | folds one tiny sequence per model; skipped unless CUDA **and** `ESMFOLD2_MODEL_DIR` / `ESMFOLD2_ESMC_DIR` / `ESMFOLD2_CCD` (or `PROTENIX_CKPT`) are set |

The count tests are the load-bearing ones: a wrong join or a policy applied at the wrong
granularity shows up as a count that no longer matches the dataset card.

## License

See `LICENSE`.
