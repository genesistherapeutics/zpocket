"""Resolve one pre-merged A3M per protein."""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from typing import Self


class MSAResolver:
    """Resolve pre-merged A3Ms, expanding gzip files into temporary storage.

    Expanded alignments live as long as the resolver, so a caller can hold one
    across a whole split and clean them up together.
    """

    def __init__(self, msa_dir: str | None, *, no_msa: bool = False) -> None:
        """
        Args:
            msa_dir: Root of the alignments, or None with `no_msa`.
            no_msa: Resolve everything to None — single-sequence folding.
        """
        if msa_dir is None and not no_msa:
            raise ValueError("provide --msa-dir or pass --no-msa")
        self.msa_dir = msa_dir
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
        assert self.msa_dir is not None
        path = os.path.join(self.msa_dir, f"{record_id}.a3m")
        if os.path.exists(path):
            return path
        if os.path.exists(path + ".gz"):
            return self._gunzip(record_id, path + ".gz")
        raise FileNotFoundError(
            f"no a3m for {record_id!r} at {path} (pass --no-msa to fold single-sequence)"
        )

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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
