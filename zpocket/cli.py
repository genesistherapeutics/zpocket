"""zpocket command-line entrypoint.

Six subcommands, in two groups. The heavyweight ESMFold2 backend (``esm``) is imported lazily
inside its dispatch, so the CLI imports and ``--help``s without that stack installed and loads
only what the chosen command needs.

**Predicting pockets** — needs a GPU and trunk weights:

- ``run`` — fold with ESMFold2 and score in one streaming pass, never persisting the quadratic
  pair rep. The command to use on your own protein, and at benchmark scale.
- ``fold`` — fold and *save* frozen-trunk features (``s`` / ``z`` / distogram / ESMC). For when
  the features themselves are the product, or you will score several head ensembles over them.
- ``predict`` — run heads over a saved ``z`` NPZ. Plain torch, CPU is fine.
- ``score-split`` — ``predict`` over a whole benchmark split's saved features.

**Measuring** — no GPU, no trunk weights, just saved score vectors:

- ``eval`` — per-target and per-pocket AUPRC, and per-centroid DCC at top-N / top-(N+2).
- ``centroids`` — rebuild the localization ground truth in your own folds' coordinate frame,
  which is what lets the benchmark be scored against folds it does not ship. The release's own
  folds already come with a matching table, so this is for folds you generated yourself.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

from zpocket.msa import COMBINE_POLICIES

if TYPE_CHECKING:  # the real import is lazy, inside the folding dispatches
    from zpocket.esmfold2.config import ESMFold2Config

_SAVE_TOKENS = {"s", "z", "esmc", "distogram"}


def parse_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(id, sequence)`` pairs from a FASTA file (first whitespace token = id)."""
    record_id: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_id is not None:
                    yield record_id, "".join(chunks)
                record_id = line[1:].split()[0] if line[1:].split() else line[1:]
                chunks = []
            else:
                chunks.append(line)
    if record_id is not None:
        yield record_id, "".join(chunks)


def _sanitize_id(record_id: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]", "_", record_id) or "input"


def _resolve_a3m(record_id: str, msa_dir: str | None, no_msa: bool) -> str | None:
    """Locate the a3m for a record, or None for single-sequence folding."""
    if no_msa:
        return None
    if msa_dir is None:
        raise SystemExit(
            "Provide precomputed MSAs via --msa-dir <dir with {id}.a3m>, or pass --no-msa "
            "for single-sequence folding."
        )
    a3m_path = os.path.join(msa_dir, f"{record_id}.a3m")
    if not os.path.exists(a3m_path):
        raise SystemExit(f"No a3m for record {record_id!r} at {a3m_path!r} (use --no-msa to skip).")
    return a3m_path


def _parse_save(save: str) -> set[str]:
    tokens = {t.strip() for t in save.split(",") if t.strip()}
    unknown = tokens - _SAVE_TOKENS
    if unknown:
        raise SystemExit(
            f"--save has unknown token(s) {sorted(unknown)}; allowed: {sorted(_SAVE_TOKENS)}"
        )
    return tokens


def _validate_save(model: str, tokens: set[str]) -> None:
    save_esmc = "esmc" in tokens
    if save_esmc and model != "esmfold2":
        raise SystemExit("--save esmc is only supported for --model esmfold2.")
    save_single = "s" in tokens
    save_pair = bool(tokens & {"z", "distogram"})
    if save_esmc and not (save_single or save_pair):
        raise SystemExit("--save esmc requires at least one of s / z (the writer's rule).")


# Recipe knobs shared by `fold` and `run`. Defaults live in ESMFold2Config (the published
# recipe); a CLI flag left at None does not override them.
_RECIPE_FLAGS = (
    "num_loops",
    "num_sampling_steps",
    "noise_scale",
    "step_scale",
    "max_inference_sigma",
    "lm_dropout",
    "msa_max_depth",
    "msa_column_mask_rate",
)


def _recipe_overrides(args: argparse.Namespace) -> dict:
    """The sampling knobs the user set explicitly, ready to pass to `ESMFold2Config`."""
    overrides = {
        name: getattr(args, name)
        for name in _RECIPE_FLAGS
        if getattr(args, name, None) is not None
    }
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "samples", None) is not None:
        overrides["num_diffusion_samples"] = args.samples
    return overrides


def _describe_recipe(cfg: ESMFold2Config) -> str:
    """One line naming every knob that shapes `z` and the sampled structure.

    Printed on every fold so a run's provenance lands in its own log, rather than being
    something to infer from which flags were passed. ``checkpoint`` means the knob was left
    unset and the checkpoint's own config governs it.
    """
    depth = cfg.msa_max_depth if cfg.msa_max_depth is not None else "checkpoint"
    col_mask = cfg.msa_column_mask_rate if cfg.msa_column_mask_rate is not None else "checkpoint"
    return (
        f"num_loops={cfg.num_loops} lm_dropout={cfg.lm_dropout} seed={cfg.seed} "
        f"samples={cfg.num_diffusion_samples} steps={cfg.num_sampling_steps} "
        f"noise={cfg.noise_scale} step_scale={cfg.step_scale} "
        f"sigma_max={cfg.max_inference_sigma} "
        f"msa_max_depth={depth} msa_col_mask={col_mask}"
    )


def _run_esmfold2(args: argparse.Namespace, tokens: set[str]) -> None:
    if not args.model_dir:
        raise SystemExit("--model-dir is required.")

    # Lazy import: pulls esm/torch, only needed for folding.
    from zpocket.esmfold2.config import ESMFold2Config
    from zpocket.esmfold2.fold import fold_protein
    from zpocket.esmfold2.loader import load_esmfold2

    cfg = ESMFold2Config(
        save_trunk_single="s" in tokens,
        save_trunk_embeddings=bool(tokens & {"z", "distogram"}),
        save_esmc_embeddings="esmc" in tokens,
        esmc_precision=args.esmc_precision,
        use_msa=not args.no_msa,
        **_recipe_overrides(args),
    )
    print(f"[fold] recipe: {_describe_recipe(cfg)}")

    model, builder = load_esmfold2(
        args.model_dir, args.esmc_dir, args.ccd, esmc_precision=args.esmc_precision
    )
    for record_id, sequence in parse_fasta(args.fasta):
        sid = _sanitize_id(record_id)
        a3m = _resolve_a3m(record_id, args.msa_dir, args.no_msa)
        out = fold_protein(
            model,
            builder,
            sequence=sequence,
            structure_id=sid,
            out_dir=args.out,
            cfg=cfg,
            model_weight_uri=args.model_dir,
            a3m_path=a3m,
        )
        print(f"[esmfold2] {sid}: trunk={out['trunk_npz']} cifs={len(out['cif_paths'])}")


def _run_predict(args: argparse.Namespace) -> None:
    # Plain-torch only: no ESMFold2 stack, no cuEquivariance.
    import numpy as np

    from zpocket.predict import load_heads, predict_from_npz

    ckpts = [c.strip() for c in args.head.split(",") if c.strip()]
    if not ckpts:
        raise SystemExit("--head requires at least one checkpoint path.")
    heads = load_heads(ckpts)
    tag = "ensemble" if len(heads) > 1 else "head"
    print(f"[predict] loaded {len(heads)} {tag} checkpoint(s): {ckpts}")

    for features_npz in args.features:
        sid = _sanitize_id(os.path.splitext(os.path.basename(features_npz))[0])
        result = predict_from_npz(features_npz, heads)
        probs = result["probs"]
        sha = result["sequence_sha256"]
        out_npz = os.path.join(args.out, f"{sid}_pocket_probs.npz")
        np.savez_compressed(
            out_npz,
            probs=probs.astype(np.float32),
            sequence_sha256=np.asarray(sha if sha is not None else ""),
        )
        out_tsv = os.path.join(args.out, f"{sid}_pocket_probs.tsv")
        with open(out_tsv, "w") as fh:
            fh.write("resnum\tprob\n")
            for i, p in enumerate(probs, start=1):
                fh.write(f"{i}\t{float(p):.6f}\n")
        print(f"[predict] {sid}: L={probs.shape[0]} npz={out_npz} tsv={out_tsv}")


def _dataset(args: argparse.Namespace):
    """The benchmark, from a local tree if given and the Hub otherwise."""
    from zpocket.data import PocketWatch

    return PocketWatch(getattr(args, "dataset_root", None))


def _split_proteins(args: argparse.Namespace, dataset):
    """The split's proteins under this run's protein filters."""
    from zpocket.data import PAPER_EXCLUDED_TEST_PROTEINS, load_split

    exclude = PAPER_EXCLUDED_TEST_PROTEINS if args.paper_set else ()
    only = None
    if getattr(args, "only_file", None):
        with open(args.only_file) as fh:
            only = [line.strip() for line in fh if line.strip()]
        if not only:
            raise SystemExit(f"{args.only_file} lists no ids")
    return load_split(
        dataset,
        args.split,
        multichain=args.multichain,
        max_length=args.max_length,
        limit=args.limit,
        exclude=exclude,
        only=only,
    )


def _run_score_split(args: argparse.Namespace) -> None:
    """Score a whole split from trunk-feature NPZs already on disk."""
    from zpocket.score import load_heads, save_scores, score_features_dir

    dataset = _dataset(args)
    proteins = _split_proteins(args, dataset)
    heads = load_heads([c.strip() for c in args.head.split(",") if c.strip()])
    print(f"[score] {len(heads)} head(s); {len(proteins)} proteins in split {args.split!r}")
    scores, missing = score_features_dir(args.features_dir, heads, [p.uniprot_id for p in proteins])
    save_scores(scores, args.out)
    print(f"[score] wrote {len(scores)} vectors -> {args.out} ({len(missing)} without features)")
    if missing:
        print(f"[score] first missing: {', '.join(missing[:5])}")


def _run_centroids(args: argparse.Namespace) -> None:
    """Rebuild the localization ground truth in the frame of a given fold set."""
    from zpocket.discrete.centroids import build_site_centroids
    from zpocket.discrete.structure import make_fold_resolver

    dataset = _dataset(args)
    lengths = dict(
        dataset.sequences().select("normalized_uniprot_id", "sequence_length").iter_rows()
    )
    resolver = make_fold_resolver(
        lengths,
        structures_dir=args.structures_dir,
        dataset=None if args.structures_dir else dataset,
    )
    frame = build_site_centroids(dataset, resolver, splits=args.splits)
    frame.write_parquet(args.out)
    print(f"[centroids] wrote {frame.height} sites -> {args.out}")
    if args.compare_to_shipped:
        shipped = dataset.site_centroids()
        joined = frame.join(shipped, on="pocket_site_id", how="inner", suffix="_ref")
        import numpy as np

        delta = np.linalg.norm(
            np.column_stack(
                [
                    joined["ligand_centroid_x"] - joined["ligand_centroid_x_ref"],
                    joined["ligand_centroid_y"] - joined["ligand_centroid_y_ref"],
                    joined["ligand_centroid_z"] - joined["ligand_centroid_z_ref"],
                ]
            ),
            axis=1,
        )
        print(
            f"[centroids] vs shipped: {joined.height} shared sites, "
            f"max {delta.max():.2e} A, median {np.median(delta):.2e} A"
        )


def _run_eval(args: argparse.Namespace) -> None:
    """Score saved per-residue predictions: per-target, per-pocket, and optionally per-centroid."""
    import json

    from zpocket.eval.report import evaluate_centroid_level, evaluate_residue_level
    from zpocket.score import load_scores

    dataset = _dataset(args)
    proteins = _split_proteins(args, dataset)
    scores = load_scores(args.scores)
    print(f"[eval] {len(scores)} score vectors; {len(proteins)} proteins in {args.split!r}")

    results = {
        "split": args.split,
        "multichain": args.multichain,
        "paper_set": bool(args.paper_set),
        "residue_level": evaluate_residue_level(scores, dataset, proteins, args.split),
    }

    if not args.no_centroid:
        from zpocket.discrete.structure import make_fold_resolver

        lengths = {p.uniprot_id: p.length for p in proteins}
        resolver = make_fold_resolver(
            lengths,
            structures_dir=args.structures_dir,
            dataset=None if args.structures_dir else dataset,
        )
        results["centroid_level"] = evaluate_centroid_level(
            scores,
            dataset,
            args.split,
            resolver,
            site_centroids=args.site_centroids,
            threshold=args.dcc_threshold,
            method=args.decode,
            score_threshold=args.score_threshold,
            link_distance=args.link_distance,
            min_residues=args.min_residues,
            bandwidth=args.bandwidth,
            restrict_to={p.uniprot_id for p in proteins},
        )

    _print_report(results)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"[eval] wrote {args.out}")


def _print_report(results: dict) -> None:
    """Print the headline numbers as a short table, so a run is readable without the JSON."""
    pt = results["residue_level"]["per_target"]
    pp = results["residue_level"]["per_pocket"]
    print("\n  per-residue")
    print(f"    AUPRC per-target      {pt['auprc_pt']:.4f}   (n={pt['n']})")
    print(f"    AUPRC pooled          {pt['auprc_pool']:.4f}")
    print(f"    AUROC per-target      {pt['auroc_pt']:.4f}")
    print(f"    R-precision           {pt['r_precision_pt']:.4f}")
    print(f"    hit@1                 {pt['hit1_pt']:.4f}")
    print(f"    random baseline       {pt['positive_fraction']:.4f}")
    print(
        f"    AUPRC per-pocket      {pp['auprc_pp']:.4f}   "
        f"({pp['n_pockets']} pockets / {pp['n_proteins']} targets)"
    )
    print(f"      single-pocket       {pp['auprc_pp_single']:.4f}")
    print(f"      multi-pocket        {pp['auprc_pp_multi']:.4f}")
    print(f"      lift over baseline  {pp['lift']:.1f}x")
    if "centroid_level" in results:
        cs = results["centroid_level"]["summary"]
        print(f"\n  per-centroid (DCC @ {cs['distance_threshold']:.1f} A)")
        print(f"    DCC top-N             {cs['dcc_top_n']:.4f}")
        print(f"    DCC top-N+2           {cs['dcc_top_n_plus_2']:.4f}")
        print(f"    DCC ceiling           {cs['dcc_ceiling']:.4f}")
        print(f"    AP                    {cs['ap']:.4f}")
        print(f"    predictions/protein   {cs['mean_predictions']:.1f}")
        print(f"    sites / targets       {cs['n_sites']} / {cs['n_proteins']}")
    print()


def _run_run(args: argparse.Namespace) -> None:
    """Fold + score in one streaming pass: never persists the quadratic pair rep.

    Works from a FASTA (your own protein) or from a benchmark split, which are the same loop over
    different sources of (id, sequence).
    """
    import numpy as np

    from zpocket.esmfold2.config import ESMFold2Config
    from zpocket.esmfold2.fold import fold_capture_z
    from zpocket.esmfold2.loader import load_esmfold2
    from zpocket.msa import MSAResolver, pocketwatch_msa_index
    from zpocket.score import load_heads, save_scores, score_z

    if bool(args.fasta) == bool(args.split_source):
        raise SystemExit("Pass exactly one of --fasta or --split.")

    heads = load_heads([c.strip() for c in args.head.split(",") if c.strip()])
    print(f"[run] {len(heads)} head(s) loaded")

    index = None
    if args.fasta:
        records = list(parse_fasta(args.fasta))
        layout = "flat"
    else:
        dataset = _dataset(args)
        proteins = _split_proteins(args, dataset)
        records = [(p.uniprot_id, p.sequence) for p in proteins]
        layout = args.msa_layout
        if layout == "pocketwatch":
            index = pocketwatch_msa_index(dataset)
    if args.shard:
        records = _take_shard(records, args.shard)
    print(f"[run] {len(records)} record(s) to fold")

    os.makedirs(args.out, exist_ok=True)
    model, builder = load_esmfold2(
        args.model_dir, args.esmc_dir, args.ccd, esmc_precision=args.esmc_precision
    )
    cfg = ESMFold2Config(use_msa=not args.no_msa, **_recipe_overrides(args))
    print(f"[run] recipe: {_describe_recipe(cfg)}")

    import torch

    scores: dict[str, np.ndarray] = {}
    failures: list[tuple[str, str]] = []
    with MSAResolver(
        args.msa_dir, layout=layout, combine=args.msa_combine, index=index, no_msa=args.no_msa
    ) as msas:
        for i, (record_id, sequence) in enumerate(records, 1):
            sid = _sanitize_id(record_id)
            a3m = None
            try:
                a3m = msas.resolve(record_id)
            except FileNotFoundError as exc:
                failures.append((record_id, f"{type(exc).__name__}: {exc}"))
                print(f"[run] {i}/{len(records)} {record_id}: FAILED no MSA: {exc}")
                continue

            # Activations scale with L^2, so consecutive proteins of very different lengths leave
            # the caching allocator holding blocks of the wrong size. Releasing them between
            # records is what keeps a long protein from failing on fragmentation rather than on
            # genuine capacity; an OOM is then retried once from a clean cache before giving up.
            probs = cifs = None
            for attempt in (1, 2):
                try:
                    _release_cuda_memory()
                    z, cifs = fold_capture_z(
                        model,
                        builder,
                        sequence=sequence,
                        structure_id=sid,
                        out_dir=args.out,
                        cfg=cfg,
                        a3m_path=a3m,
                    )
                    probs = score_z(z, heads)
                    del z
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    if attempt == 1:
                        print(
                            f"[run] {i}/{len(records)} {record_id}: OOM at L={len(sequence)}, "
                            f"retrying from a cleared cache"
                        )
                        continue
                    failures.append((record_id, f"{type(exc).__name__}: {exc}"))
                    print(
                        f"[run] {i}/{len(records)} {record_id}: FAILED OutOfMemoryError at "
                        f"L={len(sequence)} after retry"
                    )
                except Exception as exc:  # noqa: BLE001 - one bad protein must not lose the split
                    failures.append((record_id, f"{type(exc).__name__}: {exc}"))
                    print(
                        f"[run] {i}/{len(records)} {record_id}: FAILED {type(exc).__name__}: {exc}"
                    )
                    break
            if probs is None:
                continue

            scores[record_id] = probs
            _write_per_residue(args.out, sid, probs)
            print(
                f"[run] {i}/{len(records)} {record_id}: L={probs.shape[0]} "
                f"max_prob={probs.max():.3f} cif={len(cifs or [])}"
            )
            if args.scores_out and i % 25 == 0:
                save_scores(scores, args.scores_out)  # checkpoint, so a crash costs little

    if args.scores_out:
        save_scores(scores, args.scores_out)
    print(f"[run] scored {len(scores)}/{len(records)} records; {len(failures)} failed")
    for record_id, message in failures[:10]:
        print(f"[run]   failed: {record_id}: {message}")


def _release_cuda_memory() -> None:
    """Return cached CUDA blocks to the driver, so the next fold starts unfragmented.

    `torch` keeps freed activations in its caching allocator, bucketed by size. Folding a split
    walks through wildly different sequence lengths and activations go as L^2, so the blocks left
    by a short protein are the wrong shape for a long one: the allocator holds gigabytes it cannot
    reuse and the long protein fails with free memory still on the card. Observed directly on the
    test split — OOM reports carried 7-14 GiB "reserved but unallocated".

    Setting ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` in the environment reduces the
    fragmentation this works around, and the two compose.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _take_shard(records: list[tuple[str, str]], spec: str) -> list[tuple[str, str]]:
    """One shard of the work, balanced by sequence length.

    Folding cost grows superlinearly in length, so splitting the list into contiguous blocks
    would leave the shard holding the long proteins running for hours after the others finish.
    Sorting by length and dealing round-robin gives every shard a comparable mix.

    Args:
        records: ``(record_id, sequence)`` pairs.
        spec: ``"i/n"``, one-based — shard 1 of 8 is ``"1/8"``.

    Returns:
        This shard's records, longest first.

    Raises:
        SystemExit: If `spec` is malformed or out of range, since a silently-wrong shard would
            quietly drop proteins from the run.
    """
    try:
        index_text, count_text = spec.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError:
        raise SystemExit(f"--shard must look like 'i/n', got {spec!r}") from None
    if not 1 <= index <= count:
        raise SystemExit(f"--shard {spec!r}: expected 1 <= i <= n")
    ordered = sorted(records, key=lambda r: -len(r[1]))
    return ordered[index - 1 :: count]


def _write_per_residue(out_dir: str, sid: str, probs) -> tuple[str, str]:
    """Write one record's per-residue probabilities as both NPZ and a readable TSV."""
    import numpy as np

    out_npz = os.path.join(out_dir, f"{sid}_pocket_probs.npz")
    np.savez_compressed(out_npz, probs=probs.astype(np.float32))
    out_tsv = os.path.join(out_dir, f"{sid}_pocket_probs.tsv")
    with open(out_tsv, "w") as fh:
        fh.write("resnum\tprob\n")
        for i, p in enumerate(probs, start=1):
            fh.write(f"{i}\t{float(p):.6f}\n")
    return out_npz, out_tsv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zpocket", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fold = sub.add_parser("fold", help="Fold proteins from a FASTA and save trunk features.")
    fold.add_argument("--model", default="esmfold2", choices=["esmfold2"])
    fold.add_argument("--fasta", required=True, help="Input FASTA (one record per protein).")
    fold.add_argument("--out", required=True, help="Output directory for NPZs + CIFs.")
    fold.add_argument("--msa-dir", default=None, help="Dir with precomputed {id}.a3m files.")
    fold.add_argument("--no-msa", action="store_true", help="Single-sequence folding (no MSA).")
    fold.add_argument(
        "--save",
        default="s",
        help="Comma-separated features to persist: s,z,esmc,distogram (default: s).",
    )
    fold.add_argument(
        "--seed", type=int, default=None, help="Diffusion seed."
    )
    fold.add_argument("--samples", type=int, default=None, help="Diffusion samples per input.")
    fold.add_argument("--esmc-precision", default="bf16", choices=["bf16", "fp32"])
    # --- Trunk + sampling recipe. Defaults are the PUBLISHED ones (PAPER_SAMPLING), so a
    # plain invocation reproduces the features the released head reads. Any knob can be
    # overridden; None here means "use the published default".
    fold.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Recurrent trunk loops (default 10, the published value; esm's own default "
        "differs). z is recomputed every loop, so this changes the prediction itself.",
    )
    fold.add_argument(
        "--num-sampling-steps", type=int, default=None, help="Diffusion steps (default 100)."
    )
    fold.add_argument("--noise-scale", type=float, default=None, help="Default 1.003.")
    fold.add_argument("--step-scale", type=float, default=None, help="Default 1.5.")
    fold.add_argument("--max-inference-sigma", type=float, default=None, help="Default 256.")
    fold.add_argument(
        "--lm-dropout",
        type=float,
        default=None,
        help="Language-model dropout, active at inference with a fresh mask per trunk loop "
        "(default 0.25, the published value). This changes z, so it changes the prediction. "
        "NOTE: 0 does not disable dropout -- esm reads a falsy value as 'leave the "
        "checkpoint config alone', which still applies the checkpoint's own rate.",
    )
    fold.add_argument(
        "--msa-max-depth",
        type=int,
        default=None,
        help="MSA rows kept per trunk loop. Unset defers to the checkpoint's own "
        "config.msa_encoder, which is what the published features used.",
    )
    fold.add_argument(
        "--msa-column-mask-rate",
        type=float,
        default=None,
        help="Fraction of MSA columns masked. Unset defers to the checkpoint config.",
    )

    # ESMFold2 weight locations.
    fold.add_argument(
        "--model-dir", required=True, help="ESMFold2 checkpoint dir or HF repo id."
    )
    fold.add_argument(
        "--esmc-dir", default=None, help="ESMC-6B backbone dir or HF repo id. See `run --help`."
    )
    fold.add_argument(
        "--ccd", default=None, help="Path to ccd.pkl. Omit to let esm resolve it."
    )

    predict = sub.add_parser(
        "predict",
        help="Run trained pocket heads over a saved pair-rep z NPZ -> per-residue pocket probs.",
    )
    predict.add_argument(
        "--head",
        required=True,
        help=(
            "Pocket head checkpoint(s), comma-separated. Multiple heads (seed replicas) are "
            "mean-ensembled at the logit level. Pair the head with the matching trunk: a head "
            "features. The published head is genesisml/zPocket, trained on ESMFold2 z."
        ),
    )
    predict.add_argument(
        "--features",
        required=True,
        nargs="+",
        help="One or more features NPZs (key 'z', shape (L, L, C)) written by 'zpocket fold'.",
    )
    predict.add_argument(
        "--out", required=True, help="Output directory for per-residue prob NPZ + TSV."
    )

    # --- Benchmark subcommands: everything below reads the PocketWatch dataset. -------------
    def _add_dataset_flags(sp, *, with_split: bool = True) -> None:
        sp.add_argument(
            "--dataset-root",
            default=None,
            help="Local PocketWatch tree (data/, projection/, structures/). "
            "Omitted: files are pulled from the genesisml/PocketWatch dataset on the Hub.",
        )
        if with_split:
            sp.add_argument("--split", default="test", choices=["train", "validation", "test"])
            sp.add_argument(
                "--multichain",
                default="keep",
                choices=["keep", "mask", "drop"],
                help="Residue-level interface-contact policy. 'keep' is what the published "
                "checkpoints were scored under; policies are not comparable with each other.",
            )
            sp.add_argument("--max-length", type=int, default=None)
            sp.add_argument(
                "--limit", type=int, default=None, help="First N proteins, for smoke runs."
            )
            sp.add_argument(
                "--only-file",
                default=None,
                help="Restrict to the UniProt accessions listed in this file, one per line. "
                "Use it to re-run the records a previous pass failed on.",
            )
            sp.add_argument(
                "--paper-set",
                action="store_true",
                help="Drop the 4 test proteins the paper excludes, giving its 1,562 pockets "
                "over 1,275 targets.",
            )

    score = sub.add_parser(
        "score-split",
        help="Score a whole split from trunk-feature NPZs -> one scores NPZ.",
    )
    _add_dataset_flags(score)
    score.add_argument("--head", required=True, help="Head checkpoint(s), comma-separated.")
    score.add_argument(
        "--features-dir",
        required=True,
        help="Directory of <uniprot>_trunk_predictions.npz from 'zpocket fold'.",
    )
    score.add_argument("--out", required=True, help="Output scores NPZ.")

    centroids = sub.add_parser(
        "centroids",
        help="Rebuild the localization ground truth in your own folds' frame.",
    )
    _add_dataset_flags(centroids, with_split=False)
    centroids.add_argument(
        "--structures-dir",
        default=None,
        help="Directory of <uniprot>.cif[.gz] folds to project into. Omitted: the shipped "
        "ESMFold2 folds, which reproduces the shipped table.",
    )
    centroids.add_argument("--out", required=True, help="Output centroid parquet.")
    centroids.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        help="Splits to build; the projection bundle covers validation and test.",
    )
    centroids.add_argument(
        "--compare-to-shipped",
        action="store_true",
        help="Report max/median deviation from the release's own site_centroids.parquet. Only "
        "meaningful when projecting into that release's structures/, where it should be ~1e-5 A.",
    )

    ev = sub.add_parser(
        "eval",
        help="Score saved predictions: per-target + per-pocket AUPRC, and per-centroid DCC.",
    )
    _add_dataset_flags(ev)
    ev.add_argument("--scores", required=True, help="Scores NPZ keyed by UniProt accession.")
    ev.add_argument("--out", default=None, help="Write the full report as JSON.")
    ev.add_argument(
        "--structures-dir",
        default=None,
        help="Folds for the centroid metrics. Must be the frame --site-centroids is in.",
    )
    ev.add_argument(
        "--site-centroids",
        default=None,
        help="Centroid parquet from 'zpocket centroids'. Omitted: the release's own "
        "site_centroids.parquet, which pairs with its own structures/ and with nothing else. "
        "Needed only when scoring folds you generated yourself.",
    )
    ev.add_argument(
        "--no-centroid", action="store_true", help="Per-residue metrics only; needs no folds."
    )
    ev.add_argument("--dcc-threshold", type=float, default=4.0)
    # The published clustering decoder, and the setting chosen on validation and frozen before
    # test was read: weighted mean shift, tau=0.3, bandwidth 14 A, m=3.
    ev.add_argument(
        "--decode",
        default="meanshift",
        choices=["linkage", "meanshift"],
        help="Clustering decoder. 'meanshift' is the published one; 'linkage' is cheaper but "
        "scores below it and is not what the published numbers used.",
    )
    ev.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Residues at or above this seed the clustering (published: 0.3).",
    )
    ev.add_argument(
        "--link-distance", type=float, default=8.0, help="'linkage' only: CA link cutoff."
    )
    ev.add_argument(
        "--min-residues",
        type=int,
        default=3,
        help="Groups smaller than this are discarded as noise (published: 3).",
    )
    ev.add_argument(
        "--bandwidth",
        type=float,
        default=14.0,
        help="'meanshift' only: kernel radius in angstroms (published: 14). Narrow bandwidths "
        "over-fragment badly -- this is not a knob to nudge downward casually.",
    )

    run = sub.add_parser(
        "run",
        help="Fold with ESMFold2 and score in one pass, without persisting the pair rep z.",
    )
    _add_dataset_flags(run)
    # --split comes from _add_dataset_flags; `run` needs to know whether it was asked for.
    run.add_argument(
        "--fasta", default=None, help="Fold these records instead of a benchmark split."
    )
    run.add_argument(
        "--split-source",
        action="store_true",
        help="Fold the --split proteins from the benchmark rather than a FASTA.",
    )
    run.add_argument("--head", required=True, help="Head checkpoint(s), comma-separated.")
    run.add_argument("--out", required=True, help="Output directory for CIFs + per-residue probs.")
    run.add_argument("--scores-out", default=None, help="Also write one combined scores NPZ.")
    run.add_argument("--msa-dir", default=None, help="Alignment root.")
    run.add_argument("--no-msa", action="store_true", help="Single-sequence folding.")
    run.add_argument(
        "--msa-layout",
        default="flat",
        choices=["flat", "pocketwatch"],
        help="'flat' reads <dir>/<id>.a3m; 'pocketwatch' reads the shipped <hash>/<source>.a3m.gz "
        "tree via msas_test_index.parquet.",
    )
    run.add_argument(
        "--msa-combine",
        default="genesis",
        choices=list(COMBINE_POLICIES),
        help="How to reduce PocketWatch's four searches to the single alignment ESMFold2 takes. "
        "'genesis' (the default) reproduces the training-time merge the published checkpoints "
        "saw -- uniprot first with its own 8192 slot, the rest cross-deduplicated into a further "
        "8191. Naming a single database instead is an ablation, not a reproduction.",
    )
    run.add_argument(
        "--shard",
        default=None,
        help="Fold only shard i of n, as 'i/n' (one-based). Work is dealt round-robin over "
        "length-sorted records, so shards finish together. Give each shard its own "
        "--scores-out and a GPU via CUDA_VISIBLE_DEVICES, then merge the NPZs.",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--samples", type=int, default=None, help="Diffusion samples (default 1).")
    run.add_argument(
        "--paper-exact",
        action="store_true",
        help="Explicitly pin the published trunk + sampling recipe. This is now also the "
        "DEFAULT, so the flag is a no-op kept for compatibility with existing scripts.",
    )
    # --- Trunk + sampling recipe. Defaults are the PUBLISHED ones (PAPER_SAMPLING), so a
    # plain invocation reproduces the features the released head reads. Any knob can be
    # overridden; None here means "use the published default".
    run.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Recurrent trunk loops (default 10, the published value; esm's own default "
        "differs). z is recomputed every loop, so this changes the prediction itself.",
    )
    run.add_argument(
        "--num-sampling-steps", type=int, default=None, help="Diffusion steps (default 100)."
    )
    run.add_argument("--noise-scale", type=float, default=None, help="Default 1.003.")
    run.add_argument("--step-scale", type=float, default=None, help="Default 1.5.")
    run.add_argument("--max-inference-sigma", type=float, default=None, help="Default 256.")
    run.add_argument(
        "--lm-dropout",
        type=float,
        default=None,
        help="Language-model dropout, active at inference with a fresh mask per trunk loop "
        "(default 0.25, the published value). This changes z, so it changes the prediction. "
        "NOTE: 0 does not disable dropout -- esm reads a falsy value as 'leave the "
        "checkpoint config alone', which still applies the checkpoint's own rate.",
    )
    run.add_argument(
        "--msa-max-depth",
        type=int,
        default=None,
        help="MSA rows kept per trunk loop. Unset defers to the checkpoint's own "
        "config.msa_encoder, which is what the published features used.",
    )
    run.add_argument(
        "--msa-column-mask-rate",
        type=float,
        default=None,
        help="Fraction of MSA columns masked. Unset defers to the checkpoint config.",
    )
    run.add_argument("--esmc-precision", default="bf16", choices=["bf16", "fp32"])
    run.add_argument("--model-dir", required=True, help="ESMFold2 checkpoint dir or HF repo id.")
    run.add_argument(
        "--esmc-dir",
        default=None,
        help="ESMC-6B backbone dir or HF repo id. Omit on a checkpoint that bundles the "
        "backbone (biohub/ESMFold2 at 'main'); on a decoder-only checkpoint, omitting it "
        "lets esm resolve the backbone named in the config, which may hit the network.",
    )
    run.add_argument(
        "--ccd",
        default=None,
        help="Path to ccd.pkl (must be named ccd.pkl). Omit to let esm resolve the CCD "
        "dictionary itself, which may hit the network.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command in ("eval", "score-split", "centroids", "run"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        {
            "eval": _run_eval,
            "score-split": _run_score_split,
            "centroids": _run_centroids,
            "run": _run_run,
        }[args.command](args)
        return 0

    if args.command == "predict":
        os.makedirs(args.out, exist_ok=True)
        _run_predict(args)
        return 0

    if args.command != "fold":
        raise SystemExit(f"Unknown command {args.command!r}")

    tokens = _parse_save(args.save)
    _validate_save(args.model, tokens)
    os.makedirs(args.out, exist_ok=True)

    _run_esmfold2(args, tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
