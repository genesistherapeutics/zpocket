"""Resolve one a3m per protein, merging PocketWatch's four per-database searches the way the
published features were built.

ESMFold2 takes a **single** alignment per chain (`esm.utils.msa.MSA`), but the PocketWatch test
MSAs ship as *four* searches per protein — ``uniprot``, ``uniref90``, ``reduced_bfd`` and
``mgnify`` — one gzipped a3m each, in a directory named by the sequence hash. The published
checkpoints were trained on features built from a single pre-merged a3m, so reproducing their
numbers means reproducing that merge exactly.

`merge_genesis` implements that merge. It is the recipe the ``genesisml/zPocket`` model card
records under ``feature_provenance.msa.merge_recipe``, and `tests/test_msa.py` asserts byte
equality against an independent transcription of it. Four details in it are not what a naive
concatenation would do, and each one changes the alignment:

1. **The order is uniprot first** — ``uniprot, uniref90, reduced_bfd, mgnify`` — not deepest-first.
2. **UniProt is deduplicated only against itself.** Its hits never enter the cross-database `seen`
   set, so a sequence found in both UniProt and UniRef90 is kept twice at this stage.
3. **The two caps are independent.** UniProt contributes at most `MAX_UNIPROT`; everything else
   contributes at most ``MAX_TOTAL - MAX_UNIPROT``, a *fixed* 8,191 rather than whatever is left
   over. A protein with 100 UniProt hits therefore yields at most 8,292 sequences, not 16,383.
4. **A final dedup pass runs over the assembled list**, first-occurrence-wins. This is where the
   duplicates that (2) deliberately let through are resolved, in favour of the UniProt copy.

The query is the record whose header starts with ``query``, taken from the first database in
`MERGE_ORDER` that has one. Records without that prefix are hits, which is why the
convention matters: a mislabelled query would be merged in as a homolog.

Two layouts are supported:

``flat``        ``<dir>/<record_id>.a3m`` — the original zpocket contract, and what you will have
                for your own protein. No merge is involved.
``pocketwatch`` ``<dir>/<hash>/<source>.a3m.gz``, located through ``msas_test_index.parquet``,
                merged as above.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

# The published merge order. Not alphabetical and not deepest-first; uniprot leads because it
# gets its own reserved block below.
MERGE_ORDER = ("uniprot", "uniref90", "reduced_bfd", "mgnify")
MSA_SOURCES = MERGE_ORDER

# The published caps. `MAX_TOTAL` is 16383, one below the 16384 depth `zpocket.esmfold2.input`
# then applies when parsing, so the parse cap never binds first.
MAX_UNIPROT = 8192
MAX_TOTAL = 16383

# A record is the query when its header starts with this; everything else is a hit.
QUERY_PREFIX = "query"

# "genesis" reproduces the training-time merge. The single-source policies are for ablations —
# they are NOT what the published checkpoints saw.
COMBINE_POLICIES = ("genesis", *MSA_SOURCES)


def read_a3m(path: str) -> list[tuple[str, str]]:
    """Read an a3m into ``(header, aligned_sequence)`` pairs, decompressing a ``.gz``.

    Headers come back without their leading ``>``. Sequences keep their gaps and their
    lower-case insertion characters exactly as written: the merge deduplicates on this string,
    so normalising it here would merge records the published merge kept apart.

    Args:
        path: Path to an a3m, plain or gzipped.

    Returns:
        One pair per record, in file order.
    """
    opener = gzip.open if path.endswith(".gz") else open
    headers: list[str] = []
    sequences: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                headers.append(line[1:])
                sequences.append("")
            elif headers:
                sequences[-1] += line.strip()
    return list(zip(headers, sequences))


def merge_genesis(paths_by_db: dict[str, str]) -> str:
    """Merge the four per-database a3ms into the single alignment the trunks were trained on.

    See the module docstring for the four non-obvious details this preserves; changing any of
    them changes the trunk features and therefore every metric downstream.

    Args:
        paths_by_db: ``{database: a3m path}``. Databases absent from the mapping are skipped,
            which is what a protein with a failed search looks like.

    Returns:
        The merged a3m as text, query first.

    Raises:
        ValueError: If no database supplies a query record. Merging hits with no query would
            produce an alignment with nothing to align against.
    """
    query: tuple[str, str] | None = None
    uniprot: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []
    seen: set[str] = set()

    for db in MERGE_ORDER:
        path = paths_by_db.get(db)
        if path is None:
            continue
        records = read_a3m(path)
        if query is None:
            query = next((r for r in records if r[0].startswith(QUERY_PREFIX)), None)
        hits = [r for r in records if not r[0].startswith(QUERY_PREFIX)]
        if db == "uniprot":
            # Deduplicated against itself only: these never join `seen`, so a sequence also found
            # in another database survives here and is resolved by the final pass below.
            within: set[str] = set()
            for record in hits:
                if record[1] not in within:
                    within.add(record[1])
                    uniprot.append(record)
        else:
            # Cross-database dedup on the aligned string; first occurrence wins.
            for record in hits:
                if record[1] not in seen:
                    seen.add(record[1])
                    others.append(record)

    if query is None:
        raise ValueError(
            f"no record with a {QUERY_PREFIX!r} header in any of {sorted(paths_by_db)}; "
            f"without a query there is nothing for the hits to align to"
        )

    # Both caps are fixed, so `others` is bounded by MAX_TOTAL - MAX_UNIPROT whatever UniProt
    # contributed -- it does not expand to fill a short UniProt block.
    chosen = [query] + uniprot[:MAX_UNIPROT] + others[: MAX_TOTAL - MAX_UNIPROT]

    out: list[str] = []
    emitted: set[str] = set()
    for header, sequence in chosen:
        if sequence not in emitted:
            emitted.add(sequence)
            out.append(f">{header}\n{sequence}\n")
    return "".join(out)


class MSAResolver:
    """Produces a plain-a3m path per record, materializing a merged file when needed.

    Merged and gzipped alignments are written into a temporary directory that lives as long as
    the resolver, so a caller can hold one across a whole split and let it clean up at the end.
    """

    def __init__(
        self,
        msa_dir: str | None,
        *,
        layout: str = "flat",
        combine: str = "genesis",
        index=None,
        no_msa: bool = False,
    ) -> None:
        """
        Args:
            msa_dir: Root of the alignments, or None with `no_msa`.
            layout: ``"flat"`` for ``<dir>/<id>.a3m``, ``"pocketwatch"`` for the shipped tree.
            combine: How to reduce PocketWatch's four searches to one alignment; one of
                `COMBINE_POLICIES`. ``"genesis"`` reproduces the training-time merge; naming a
                single database is an ablation. Ignored by the flat layout.
            index: ``{record_id: subdirectory}`` for the pocketwatch layout, from
                ``msas_test_index.parquet``.
            no_msa: Resolve everything to None — single-sequence folding.

        Raises:
            ValueError: On an unknown `layout` or `combine`, or a pocketwatch layout with no index.
        """
        if layout not in ("flat", "pocketwatch"):
            raise ValueError(f"unknown layout {layout!r}; expected 'flat' or 'pocketwatch'")
        if combine not in COMBINE_POLICIES:
            raise ValueError(f"unknown combine {combine!r}; expected one of {COMBINE_POLICIES}")
        if layout == "pocketwatch" and index is None and not no_msa:
            raise ValueError("the pocketwatch layout needs an index mapping record ids to folders")
        self.msa_dir = msa_dir
        self.layout = layout
        self.combine = combine
        self.index = index or {}
        self.no_msa = no_msa
        self._tmp: str | None = None

    def _scratch(self) -> str:
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix="zpocket_msa_")
        return self._tmp

    def resolve(self, record_id: str) -> str | None:
        """Path to a plain a3m for one record, or None for single-sequence folding.

        Args:
            record_id: FASTA record id, or UniProt accession for a split run.

        Returns:
            A readable ``.a3m`` path, or None when `no_msa` is set.

        Raises:
            FileNotFoundError: If the record has no alignment. Silently folding it
                single-sequence would put one protein on different inputs from the rest.
        """
        if self.no_msa:
            return None
        if self.layout == "flat":
            return self._resolve_flat(record_id)
        return self._resolve_pocketwatch(record_id)

    def _resolve_flat(self, record_id: str) -> str:
        path = os.path.join(self.msa_dir, f"{record_id}.a3m")
        if os.path.exists(path):
            return path
        if os.path.exists(path + ".gz"):
            return self._gunzip(record_id, path + ".gz")
        raise FileNotFoundError(
            f"no a3m for {record_id!r} at {path} (pass --no-msa to fold single-sequence)"
        )

    def _resolve_pocketwatch(self, record_id: str) -> str:
        folder = self.index.get(record_id)
        if folder is None:
            raise FileNotFoundError(f"{record_id!r} is not in the MSA index")
        # The index spells the folder with its `msas_test/` prefix, so `msa_dir` is the directory
        # the tar was extracted *into*. Accept a pointer at `msas_test/` itself too, since that
        # is the more obvious reading of "the MSA directory".
        root = os.path.join(self.msa_dir, folder)
        if not os.path.isdir(root):
            alternative = os.path.join(self.msa_dir, os.path.basename(folder.rstrip("/")))
            if os.path.isdir(alternative):
                root = alternative
            else:
                raise FileNotFoundError(
                    f"no alignment folder for {record_id!r}: tried {root} and {alternative}. "
                    f"--msa-dir should be the directory msas_test.tar was extracted into."
                )
        wanted = MSA_SOURCES if self.combine == "genesis" else (self.combine,)

        paths_by_db: dict[str, str] = {}
        for source in wanted:
            for candidate in (f"{source}.a3m.gz", f"{source}.a3m"):
                full = os.path.join(root, candidate)
                if os.path.exists(full):
                    paths_by_db[source] = full
                    break
        if not paths_by_db:
            raise FileNotFoundError(f"no {list(wanted)} alignment under {root}")
        if len(paths_by_db) < len(wanted):
            logger.warning(
                "%s: found %d of %d requested alignments (%s)",
                record_id,
                len(paths_by_db),
                len(wanted),
                ", ".join(sorted(set(wanted) - set(paths_by_db))),
            )

        out = os.path.join(self._scratch(), f"{record_id}.a3m")
        if os.path.exists(out):
            return out
        if self.combine == "genesis":
            text = merge_genesis(paths_by_db)
        else:
            # A single database, passed through unmerged but still capped the way the merge caps
            # the non-UniProt block, so an ablation differs only in which hits it saw.
            records = read_a3m(next(iter(paths_by_db.values())))
            text = "".join(f">{h}\n{s}\n" for h, s in records[:MAX_TOTAL])
        with open(out, "w") as handle:
            handle.write(text)
        return out

    def _gunzip(self, record_id: str, path: str) -> str:
        out = os.path.join(self._scratch(), f"{record_id}.a3m")
        if not os.path.exists(out):
            with gzip.open(path, "rt") as src, open(out, "w") as dst:
                shutil.copyfileobj(src, dst)
        return out

    def cleanup(self) -> None:
        """Remove any alignments this resolver materialized."""
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __enter__(self) -> MSAResolver:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def pocketwatch_msa_index(dataset) -> dict[str, str]:
    """``{uniprot_id: msa subdirectory}`` from the benchmark's MSA index.

    Args:
        dataset: A `zpocket.data.PocketWatch`.

    Returns:
        The mapping named by ``msas_test_index.parquet``.
    """
    import polars as pl

    frame = pl.read_parquet(dataset.path("msas_test_index.parquet"))
    column = "msa_dir" if "msa_dir" in frame.columns else "sequence_sha256"
    return dict(frame.select("normalized_uniprot_id", column).iter_rows())
