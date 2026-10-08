"""The published MSA merge, pinned against an independent transcription of it.

The merged a3m is what the trunk actually sees, so a wrong merge moves every metric while
everything still runs. `_reference_merge` below is the recipe the ``genesisml/zPocket`` model
card records (``feature_provenance.msa.merge_recipe``) written out independently, and
`test_matches_reference_on_random_inputs` asserts byte equality against it on randomized
inputs — including the sizes where the two caps and the deliberate UniProt-duplicate
behaviour actually bite.

The named tests around it document *why* each behaviour is there, so a future change that breaks
one gets a reason rather than just a diff.
"""

from __future__ import annotations

import gzip
import os
import random

import pytest

from zpocket.msa import (
    MAX_TOTAL,
    MAX_UNIPROT,
    MERGE_ORDER,
    merge_genesis,
    read_a3m,
)

# --- the reference, transcribed verbatim -------------------------------------------------------

ORDER = ["uniprot", "uniref90", "reduced_bfd", "mgnify"]
_REF_MAX_UNIPROT, _REF_MAX_TOTAL = 8192, 16383


def _ref_read_a3m(path):
    headers, seqs = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                headers.append(line[1:])
                seqs.append("")
            elif headers:
                seqs[-1] += line.strip()
    return list(zip(headers, seqs))


def _reference_merge(paths_by_db):
    query, uniprot, others, seen = None, [], [], set()
    for db in ORDER:
        records = _ref_read_a3m(paths_by_db[db])
        if query is None:
            query = next((r for r in records if r[0].startswith("query")), None)
        hits = [r for r in records if not r[0].startswith("query")]
        if db == "uniprot":
            u = set()
            uniprot = [r for r in hits if not (r[1] in u or u.add(r[1]))]
        else:
            for r in hits:  # cross-database dedup on the aligned string; first occurrence wins
                if r[1] not in seen:
                    seen.add(r[1])
                    others.append(r)
    chosen = [query] + uniprot[:_REF_MAX_UNIPROT] + others[: _REF_MAX_TOTAL - _REF_MAX_UNIPROT]
    out, s = [], set()
    for h, q in chosen:
        if q not in s:
            s.add(q)
            out.append(f">{h}\n{q}\n")
    return "".join(out)


# --- fixtures ---------------------------------------------------------------------------------

QUERY = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


def _write(path, records):
    """Write ``(header, sequence)`` records as a gzipped a3m."""
    with gzip.open(path, "wt") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def _random_bank(rng, n, length=len(QUERY)):
    """`n` aligned strings drawn from a small alphabet, so collisions happen often."""
    return ["".join(rng.choice("ACDEFG-") for _ in range(length)) for _ in range(n)]


def _build_case(tmp_path, rng, counts, *, shared_pool=None, query_in=MERGE_ORDER):
    """Four a3ms with the given hit counts, drawing from a shared pool to force duplicates."""
    pool = shared_pool if shared_pool is not None else _random_bank(rng, 400)
    paths = {}
    for db in MERGE_ORDER:
        records = []
        if db in query_in:
            records.append((f"query_{db}", QUERY))
        for i in range(counts[db]):
            records.append((f"{db}_hit_{i}", rng.choice(pool)))
        path = tmp_path / f"{db}.a3m.gz"
        _write(path, records)
        paths[db] = str(path)
    return paths


# --- the equivalence test ---------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_matches_reference_on_random_inputs(tmp_path, seed):
    """Byte-identical to the reference on randomized inputs with heavy cross-database overlap."""
    rng = random.Random(seed)
    counts = {db: rng.randint(0, 60) for db in MERGE_ORDER}
    paths = _build_case(tmp_path, rng, counts)
    assert merge_genesis(paths) == _reference_merge(paths)


def test_matches_reference_when_a_cap_binds(tmp_path):
    """At the caps the two blocks are truncated independently; equality must still hold.

    Uses distinct (non-colliding) sequences so the counts reach the caps rather than dedup away.
    """
    rng = random.Random(99)
    paths = {}
    for db, count in (
        ("uniprot", MAX_UNIPROT + 50),
        ("uniref90", (MAX_TOTAL - MAX_UNIPROT) + 50),
        ("reduced_bfd", 10),
        ("mgnify", 10),
    ):
        records = [(f"query_{db}", QUERY)]
        records += [(f"{db}_{i}", f"{db}{i:06d}" + "-" * 10) for i in range(count)]
        path = tmp_path / f"{db}.a3m.gz"
        _write(path, records)
        paths[db] = str(path)
    merged = merge_genesis(paths)
    assert merged == _reference_merge(paths)
    # Query + both capped blocks, and nothing beyond MAX_TOTAL.
    assert merged.count(">") == 1 + MAX_UNIPROT + (MAX_TOTAL - MAX_UNIPROT)
    assert merged.count(">") <= MAX_TOTAL + 1
    del rng


# --- what each behaviour is, documented -------------------------------------------------------


def test_uniprot_comes_first(tmp_path):
    """The merge order is uniprot-first, so its hits outrank every other database's."""
    rng = random.Random(1)
    paths = _build_case(tmp_path, rng, dict.fromkeys(MERGE_ORDER, 3))
    merged = merge_genesis(paths)
    headers = [line[1:] for line in merged.splitlines() if line.startswith(">")]
    assert headers[0].startswith("query")
    first_hit_dbs = [h.split("_hit_")[0] for h in headers[1:]]
    assert first_hit_dbs[0] == "uniprot", headers[:4]


def test_others_cap_does_not_expand_when_uniprot_is_short(tmp_path):
    """The non-UniProt block is capped at a fixed 8,191, not at whatever UniProt left unused.

    A naive "fill to MAX_TOTAL" implementation would emit thousands more sequences here.
    """
    paths = {}
    for db, count in (
        ("uniprot", 5),
        ("uniref90", MAX_TOTAL),
        ("reduced_bfd", 0),
        ("mgnify", 0),
    ):
        records = [(f"query_{db}", QUERY)]
        records += [(f"{db}_{i}", f"{db}{i:06d}" + "-" * 10) for i in range(count)]
        path = tmp_path / f"{db}.a3m.gz"
        _write(path, records)
        paths[db] = str(path)
    merged = merge_genesis(paths)
    assert merged == _reference_merge(paths)
    # 1 query + 5 uniprot + 8191 others, NOT 1 + 5 + 16377.
    assert merged.count(">") == 1 + 5 + (MAX_TOTAL - MAX_UNIPROT)


def test_a_sequence_in_both_uniprot_and_uniref90_is_emitted_once_as_uniprot(tmp_path):
    """UniProt skips the cross-database dedup, and the final pass resolves it in UniProt's favour.

    The duplicate is therefore kept once, under its *UniProt* header — and it still consumes a
    slot in the non-UniProt block, which is the observable consequence of doing it this way.
    """
    shared = "ACDEFACDEF" + "-" * 10
    paths = {}
    for db, records in (
        ("uniprot", [("query_u", QUERY), ("uniprot_shared", shared)]),
        ("uniref90", [("query_r", QUERY), ("uniref90_shared", shared)]),
        ("reduced_bfd", [("query_b", QUERY)]),
        ("mgnify", [("query_m", QUERY)]),
    ):
        path = tmp_path / f"{db}.a3m.gz"
        _write(path, records)
        paths[db] = str(path)
    merged = merge_genesis(paths)
    assert merged == _reference_merge(paths)
    headers = [line[1:] for line in merged.splitlines() if line.startswith(">")]
    assert "uniprot_shared" in headers
    assert "uniref90_shared" not in headers
    assert merged.count(shared) == 1


def test_query_is_taken_from_the_first_database_that_has_one(tmp_path):
    """Order decides whose query header survives; uniprot's wins when present."""
    rng = random.Random(2)
    paths = _build_case(tmp_path, rng, dict.fromkeys(MERGE_ORDER, 2))
    headers = [line[1:] for line in merge_genesis(paths).splitlines() if line.startswith(">")]
    assert headers[0] == "query_uniprot"

    # With uniprot carrying no query, the next database in order supplies it.
    second = tmp_path / "b"
    second.mkdir()
    paths2 = _build_case(
        second,
        random.Random(2),
        dict.fromkeys(MERGE_ORDER, 2),
        query_in=("uniref90", "reduced_bfd", "mgnify"),
    )
    headers2 = [line[1:] for line in merge_genesis(paths2).splitlines() if line.startswith(">")]
    assert headers2[0] == "query_uniref90"


def test_a_hit_equal_to_the_query_is_dropped(tmp_path):
    """The query is emitted first, so the final dedup pass removes any hit identical to it."""
    paths = {}
    for db in MERGE_ORDER:
        records = [(f"query_{db}", QUERY), (f"{db}_echo", QUERY)]
        path = tmp_path / f"{db}.a3m.gz"
        _write(path, records)
        paths[db] = str(path)
    merged = merge_genesis(paths)
    assert merged == _reference_merge(paths)
    assert merged == f">query_uniprot\n{QUERY}\n"


def test_missing_databases_are_skipped(tmp_path):
    """A failed search is a missing file, not a failure: merge what is there.

    The reference indexes `paths_by_db[db]` directly and would raise; zpocket skips, so a protein
    whose mgnify search failed still folds. Nothing else about the merge changes.
    """
    records = [("query_u", QUERY), ("u_1", "ACDEF" + "-" * 28)]
    path = tmp_path / "uniprot.a3m.gz"
    _write(path, records)
    merged = merge_genesis({"uniprot": str(path)})
    assert merged.count(">") == 2


def test_no_query_anywhere_raises(tmp_path):
    """Hits with nothing to align against are not a usable alignment."""
    path = tmp_path / "uniprot.a3m.gz"
    _write(path, [("just_a_hit", QUERY)])
    with pytest.raises(ValueError, match="no record with a 'query' header"):
        merge_genesis({"uniprot": str(path)})


def test_read_a3m_joins_wrapped_lines_and_keeps_insertion_case(tmp_path):
    """Dedup is on the aligned string, so wrapping must be joined and case must be preserved."""
    path = tmp_path / "w.a3m"
    path.write_text(">q\nMKTA\nyiak\n>h\nMKTAYIAK\n")
    assert read_a3m(str(path)) == [("q", "MKTAyiak"), ("h", "MKTAYIAK")]


# --- against the real shipped alignments -------------------------------------------------------

_REAL_MSA_ROOT = os.environ.get("POCKETWATCH_MSA_ROOT")


@pytest.mark.skipif(not _REAL_MSA_ROOT, reason="POCKETWATCH_MSA_ROOT not set")
def test_matches_reference_on_real_shipped_msas(dataset):
    """Byte-identical to the reference on the alignments actually shipped with the benchmark.

    The synthetic cases above cover the branches; this covers the real thing — depths in the
    tens of thousands, real headers, and the caps genuinely binding.
    """
    from zpocket.msa import pocketwatch_msa_index

    index = pocketwatch_msa_index(dataset)
    checked = 0
    for uid in sorted(index)[:5]:
        folder = os.path.join(_REAL_MSA_ROOT, index[uid])
        if not os.path.isdir(folder):
            continue
        paths = {s: os.path.join(folder, f"{s}.a3m.gz") for s in MERGE_ORDER}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        assert merge_genesis(paths) == _reference_merge(paths), uid
        checked += 1
    if not checked:
        pytest.skip(f"no alignment folders found under {_REAL_MSA_ROOT}")


@pytest.mark.skipif(not _REAL_MSA_ROOT, reason="POCKETWATCH_MSA_ROOT not set")
def test_real_merged_query_is_the_canonical_sequence(dataset):
    """The merged query must be the sequence the labels and folds are indexed against.

    If it were not, every residue score would be aligned to the wrong position.
    """
    import polars as pl

    from zpocket.msa import MSAResolver, pocketwatch_msa_index

    index = pocketwatch_msa_index(dataset)
    sequences = dict(dataset.sequences().select("normalized_uniprot_id", "sequence").iter_rows())
    uid = next(
        (u for u in sorted(index) if os.path.isdir(os.path.join(_REAL_MSA_ROOT, index[u]))), None
    )
    if uid is None:
        pytest.skip(f"no alignment folders found under {_REAL_MSA_ROOT}")
    with MSAResolver(_REAL_MSA_ROOT, layout="pocketwatch", index=index) as msas:
        lines = open(msas.resolve(uid)).read().splitlines()
    assert lines[0].startswith(">query")
    assert lines[1] == sequences[uid]
    del pl
