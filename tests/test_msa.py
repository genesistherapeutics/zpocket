"""Resolution of plain and gzipped pre-merged A3Ms."""

from __future__ import annotations

import gzip

import pytest

from zpocket.msa import MSAResolver


def test_resolves_plain_a3m_without_copying(tmp_path):
    path = tmp_path / "protein.a3m"
    path.write_text(">query\nACDE\n")
    with MSAResolver(str(tmp_path)) as resolver:
        assert resolver.resolve("protein") == str(path)


def test_expands_gzipped_a3m(tmp_path):
    compressed = tmp_path / "protein.a3m.gz"
    with gzip.open(compressed, "wt") as handle:
        handle.write(">query\nACDE\n")

    with MSAResolver(str(tmp_path)) as resolver:
        resolved = resolver.resolve("protein")
        assert resolved is not None
        with open(resolved) as handle:
            assert handle.read() == ">query\nACDE\n"


def test_missing_a3m_is_an_error(tmp_path):
    with (
        MSAResolver(str(tmp_path)) as resolver,
        pytest.raises(FileNotFoundError, match="no a3m"),
    ):
        resolver.resolve("missing")


def test_no_msa_mode_needs_no_directory():
    with MSAResolver(None, no_msa=True) as resolver:
        assert resolver.resolve("protein") is None


def test_directory_is_required_unless_msa_is_disabled():
    with pytest.raises(ValueError, match="provide --msa-dir"):
        MSAResolver(None)
