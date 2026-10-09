"""Build a single-chain ESMFold2 ``StructurePredictionInput`` from local inputs.

Reads a local a3m file; the serializable ``esmfold2_input`` dict this produces is
the same shape ``single_chain_protein_metadata`` consumes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from esm.utils.msa import MSA
from esm.utils.structure.input_builder import (
    StructurePredictionInput,
    deserialize_structure_prediction_input,
)

# Cap MSA depth fed to ESMFold2. This is the depth the published features were built at.
MAX_MSA_SEQS = 16384


def build_esmfold2_input_dict(sequence: str, chain_id: str = "A") -> dict[str, Any]:
    """The serializable single-chain protein input dict (esm SPI schema).

    Kept in the exact shape ``deserialize_structure_prediction_input`` and
    ``single_chain_protein_metadata`` both consume.
    """
    return {"sequences": [{"type": "protein", "id": chain_id, "sequence": sequence}]}


def msa_from_local_a3m(a3m_path: str, *, zero_deletions: bool = False) -> MSA:
    """Parse a local a3m file into an esm ``MSA`` object.

    The historical esm parser discarded insertion counts after removing lowercase
    residues. ``zero_deletions=True`` reproduces that behavior for the published
    zPocket checkpoint; False preserves modern esm's deletion features.
    """
    msa = MSA.from_a3m(a3m_path, remove_insertions=True, max_sequences=MAX_MSA_SEQS)
    if zero_deletions:
        msa = dataclasses.replace(msa, deletions=None)
    return msa


def build_spi(
    esmfold2_input: dict[str, Any],
    *,
    a3m_path: str | None = None,
    zero_msa_deletions: bool = False,
) -> StructurePredictionInput:
    """Rebuild a ``StructurePredictionInput`` and inject a local MSA if provided.

    esm's deserializer leaves each protein's ``msa=None``; when ``a3m_path`` is
    given, the parsed MSA is injected into the (single) protein chain. Passing
    ``a3m_path=None`` yields single-sequence folding.
    """
    spi = deserialize_structure_prediction_input(esmfold2_input)
    if a3m_path is not None:
        # deserialize preserves sequence order 1:1 with the input dict; a single
        # protein chain is the only supported case here.
        for spi_seq, raw in zip(spi.sequences, esmfold2_input["sequences"], strict=True):
            if raw.get("type") == "protein":
                spi_seq.msa = msa_from_local_a3m(a3m_path, zero_deletions=zero_msa_deletions)
    return spi
