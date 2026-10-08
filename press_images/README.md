# Press images

## `paper_figure1.png`

Figure 1 of the zPocket paper, shown in the main README under the introduction. It is exported from the paper's PDF at 320 dpi without its caption, so regenerate it from the paper rather than editing it here. Panels B and C use the ESMFold2 trunk and `pairformer2` head that this repository ships.

## `pocket_id_comparison.{png,pdf,svg}`

Pocket-identification AUPRC for zPocket alongside the cofolding-based pocket predictors in the IsoDDE and AnewDDE technical reports. Each bar shows the number its developers reported, with P2Rank as the shared reference point where one was reported.

The tag under zPocket marks that its result can be reproduced: the PocketWatch test set ([huggingface.co/datasets/genesisml/PocketWatch](https://huggingface.co/datasets/genesisml/PocketWatch)) and the zPocket weights ([huggingface.co/genesisml/zPocket](https://huggingface.co/genesisml/zPocket)) are released with the paper. N = 1,275 is the number of test proteins scored (1,562 pockets), from the paper's Table 1. The IsoDDE report does not give the size of its pocket-identification test set.

Regenerate with `python make_pocket_id_comparison.py` (needs matplotlib only).

| Model | AUPRC | P2Rank | Source |
|---|---|---|---|
| zPocket (ESMFold2 trunk, `pairformer2` head) | 0.745 | 0.477 | zPocket paper, Table 1: per-target AUPRC on the PocketWatch test set (1,275 targets, 1,562 pockets) |
| IsoDDE | 0.75 | 0.51 | IsoDDE technical report (Isomorphic Labs, Feb 2026), Fig. 11 (left) |
| AnewDDE | 0.751 | not reported | [AnewDDE announcement](https://anewbt.com/news/anewdde-agentic-drug-discovery-engine/) (Anew Labs, Sep 2026): "AnewFold reaches an overall AUPRC of 0.751, including 0.663 for cryptic pockets and 0.755 for non-cryptic pockets" |

### Caveats

- **Different test sets.** Each model is scored on its own group's temporally held-out set, with different cutoff dates, proteins and training data. The values are approximate comparisons, not a head-to-head benchmark.
- **Shared protocol choices (zPocket and IsoDDE).** Both define pocket residues as those within 5 Å of a ligand heavy atom, keep ligands with 6–50 heavy atoms, and cluster pocket instances by ligand centroid at 5 Å. Both ran P2Rank with its AlphaFold configuration on their own predicted structures. The AnewDDE announcement gives no protocol details. Its full technical report PDF has not been checked.
- **Aggregation.** zPocket reports per-target AUPRC (one score per protein, averaged). The IsoDDE report does not say whether its AUPRC is per protein, per pocket or pooled.
- **Random baselines are not comparable.** IsoDDE's random baseline (0.16) scores residues near randomly placed pocket centres. zPocket's (0.049) is the positive-residue rate. That is why the figure omits them.
- **Crypticity split omitted.** IsoDDE reports 0.76 not-cryptic and 0.63 cryptic. zPocket reports 0.63 and 0.55 per-pocket on 423 and 40 pockets. The cryptic definitions and aggregation differ, so we left this split out rather than show it out of context.
