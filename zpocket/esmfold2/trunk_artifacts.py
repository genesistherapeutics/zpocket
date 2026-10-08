"""Import-light helpers for persisting ESMFold2 trunk prediction artifacts.

numpy + hashlib only. Schema version 3 and the payload layout match the format the
published features were distributed in, so artifacts written here are interchangeable
with them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

TRUNK_ARTIFACT_SCHEMA_VERSION = 3
ESMFOLD2_MODEL_NAME = "ESMFold2"
PROTEIN_MOL_TYPE = 0


def single_chain_protein_metadata(esmfold2_input: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build the token-to-residue mapping for a single protein chain.

    Args:
        esmfold2_input: Serialized ESMFold2 ``StructurePredictionInput``.

    Returns:
        NumPy arrays mapping every ESMFold2 token to its chain and residue.

    Raises:
        ValueError: If the input is not exactly one single-copy protein chain.
    """
    sequences = esmfold2_input.get("sequences", [])
    if len(sequences) != 1 or sequences[0].get("type") != "protein":
        raise ValueError("trunk prediction saving currently requires one protein sequence")

    protein = sequences[0]
    chain_ids = protein.get("id")
    if isinstance(chain_ids, str):
        chain_id = chain_ids
    elif isinstance(chain_ids, list) and len(chain_ids) == 1:
        chain_id = str(chain_ids[0])
    else:
        raise ValueError("trunk prediction saving requires exactly one protein chain ID")

    sequence = str(protein.get("sequence", ""))
    if not sequence:
        raise ValueError("trunk prediction saving requires a non-empty protein sequence")

    length = len(sequence)
    return {
        "token_index": np.arange(length, dtype=np.int32),
        "residue_index": np.arange(length, dtype=np.int32),
        "residue_number": np.arange(1, length + 1, dtype=np.int32),
        "chain_id": np.full(length, chain_id, dtype=f"<U{max(1, len(chain_id))}"),
        "residue_one_letter": np.asarray(list(sequence), dtype="<U1"),
        "mol_type": np.full(length, PROTEIN_MOL_TYPE, dtype=np.int32),
        "pad_token_mask": np.ones(length, dtype=np.float32),
        "sequence": np.asarray(sequence),
        "sequence_sha256": np.asarray(hashlib.sha256(sequence.encode()).hexdigest()),
    }


def write_trunk_artifact(
    output_path: str | Path,
    *,
    esmfold2_input: dict[str, Any],
    structure_id: str,
    structure_seed: int,
    model_weight_uri: str,
    s: np.ndarray | None = None,
    z: np.ndarray | None = None,
    pdistogram: np.ndarray | None = None,
    distogram_bin_boundaries_angstrom: np.ndarray | None = None,
    esmc: np.ndarray | None = None,
) -> None:
    """Persist a seed's per-residue and/or pair trunk representations as fp16.

    Casting happens here, after folding has completed, so persistence cannot alter
    structure generation or confidence calculations. At least one of ``s`` / ``z``
    must be supplied; the pair block (``z`` + ``pdistogram`` + boundaries) and the
    single block (``s``) are each written only when provided, so callers can emit an
    s-only artifact that skips the O(N^2) pair tensors. Additional optional
    per-residue arrays can be added later by following the same provided-only pattern.

    Args:
        output_path: Destination NPZ path.
        esmfold2_input: Serialized single-chain protein input.
        structure_id: Structure identifier associated with the prediction.
        structure_seed: Seed used for this trunk and its diffusion samples.
        model_weight_uri: Configured ESMFold2 model-weight location.
        s: Optional per-residue single representation with shape ``(N, C_s)``.
        z: Optional unbatched pair representation with shape ``(N, N, C_z)``.
        pdistogram: Optional unbatched raw distogram logits with shape ``(N, N, B)``.
            Required when ``z`` is given.
        distogram_bin_boundaries_angstrom: The ``B - 1`` distance boundaries.
            Required when ``z`` is given.
        esmc: Optional ESMC-6B backbone per-residue embedding (last layer only) with
            shape ``(N, C_esmc)``. Supplementary — does not on its own satisfy the
            "at least one of s / z" requirement.

    Raises:
        ValueError: If neither ``s`` nor ``z`` is given, the pair block is
            incomplete, or a tensor's shape does not agree with the residue mapping.
    """
    if s is None and z is None:
        raise ValueError("write_trunk_artifact requires at least one of s / z")

    metadata = single_chain_protein_metadata(esmfold2_input)
    length = len(metadata["token_index"])

    payload: dict[str, np.ndarray] = {
        **metadata,
        "structure_id": np.asarray(structure_id),
        "structure_seed": np.asarray(structure_seed, dtype=np.int64),
        "model_name": np.asarray(ESMFOLD2_MODEL_NAME),
        "model_weight_uri": np.asarray(model_weight_uri),
        "tensor_dtype": np.asarray("float16"),
        "schema_version": np.asarray(TRUNK_ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
    }

    if s is not None:
        if s.ndim != 2 or s.shape[0] != length:
            raise ValueError(f"s must have shape ({length}, C_s); got {s.shape}")
        payload["s"] = np.asarray(s, dtype=np.float16)

    if z is not None:
        if pdistogram is None or distogram_bin_boundaries_angstrom is None:
            raise ValueError("z requires pdistogram and distogram_bin_boundaries_angstrom")
        if z.ndim != 3 or z.shape[:2] != (length, length):
            raise ValueError(f"z must have shape ({length}, {length}, C); got {z.shape}")
        if pdistogram.ndim != 3 or pdistogram.shape[:2] != (length, length):
            raise ValueError(
                f"pdistogram must have shape ({length}, {length}, B); got {pdistogram.shape}"
            )
        if len(distogram_bin_boundaries_angstrom) != pdistogram.shape[-1] - 1:
            raise ValueError(
                "distogram boundaries must have one fewer element than the logits' bin dimension"
            )
        payload["z"] = np.asarray(z, dtype=np.float16)
        payload["pdistogram"] = np.asarray(pdistogram, dtype=np.float16)
        payload["distogram_bin_boundaries_angstrom"] = np.asarray(
            distogram_bin_boundaries_angstrom, dtype=np.float32
        )

    if esmc is not None:
        if esmc.ndim != 2 or esmc.shape[0] != length:
            raise ValueError(f"esmc must have shape ({length}, C_esmc); got {esmc.shape}")
        payload["esmc"] = np.asarray(esmc, dtype=np.float16)
        # Provenance: only the final ESMC layer is stored, out of the full stack the
        # trunk mixes over.
        payload["esmc_layer_index"] = np.asarray(-1, dtype=np.int32)
        payload["esmc_num_layers_available"] = np.asarray(81, dtype=np.int32)

    np.savez_compressed(output_path, **payload)
