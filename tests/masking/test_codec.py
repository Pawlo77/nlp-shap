"""Tests for packed mask encoding and hashing."""

import numpy as np
import pytest

from nlp_shap.masking.codec import MaskCodec, PackedMask

LEGACY_TOY_MASKS: tuple[tuple[bool, ...], ...] = (
    (True, False, True, True, False, False, True),
    (True, False, True, False),
)


def test_pack_unpack_roundtrip() -> None:
    """Packed masks round-trip to the original boolean sequence."""
    mask = LEGACY_TOY_MASKS[0]
    packed = MaskCodec.pack(mask)
    restored = MaskCodec.unpack(packed)
    assert restored == mask


@pytest.mark.parametrize("mask", LEGACY_TOY_MASKS)
def test_hash_stable_across_shapes(mask: tuple[bool, ...]) -> None:
    """Row-shaped and flat masks hash to the same packed value."""
    row = np.asarray([mask], dtype=np.bool_)
    flat = np.asarray(mask, dtype=np.bool_)
    assert MaskCodec.hash_mask(row) == MaskCodec.hash_mask(flat)


@pytest.mark.parametrize("mask", LEGACY_TOY_MASKS)
def test_hash_matches_legacy_pack_bytes(mask: tuple[bool, ...]) -> None:
    """Packed bytes and hash match the 0.x mask codec behaviour."""
    packed = MaskCodec.pack(mask)
    legacy_packed = _legacy_pack(mask)
    assert packed.words == legacy_packed.words
    assert packed.n_bits == legacy_packed.n_bits
    assert MaskCodec.hash_mask(mask) == hash((packed.words, packed.n_bits))


def test_normalize_rejects_multi_row_masks() -> None:
    """Only single-row 2D masks are accepted."""
    invalid = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    with pytest.raises(ValueError, match="exactly one row"):
        MaskCodec.normalize(invalid)


def _legacy_pack(mask: tuple[bool, ...]) -> PackedMask:
    """Reference packer mirroring MLLM-Shap ``MaskCodec.pack``."""
    bits = np.asarray(mask, dtype=np.uint8)
    packed = np.packbits(bits, bitorder="little")
    return PackedMask(words=packed.tobytes(), n_bits=len(mask))
