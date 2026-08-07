from __future__ import annotations

import hashlib
import mmap
import struct
import unicodedata
from pathlib import Path

MAGIC = b"CMSFXBL1"
HEADER = struct.Struct("<8sQIQ")


def normalize_suffix_form(word: str) -> str:
    return (
        unicodedata.normalize("NFC", str(word).strip().lower())
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
    )


def bloom_positions(value: str, bit_count: int, hash_count: int):
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=16, person=b"cmsuffix"
    ).digest()
    first, second = struct.unpack("<QQ", digest)
    second |= 1
    mask = bit_count - 1
    for index in range(hash_count):
        yield (first + index * second) & mask


class SuffixBloomIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Suffix Bloom index not found: {self.path}")
        self._handle = self.path.open("rb")
        self._mapping = mmap.mmap(
            self._handle.fileno(), length=0, access=mmap.ACCESS_READ
        )
        magic, self.bit_count, self.hash_count, self.generated_count = (
            HEADER.unpack_from(self._mapping, 0)
        )
        if magic != MAGIC:
            raise ValueError(f"Invalid suffix Bloom index: {self.path}")
        self._offset = HEADER.size

    def contains(self, surface: str) -> bool:
        value = normalize_suffix_form(surface)
        if not value:
            return False
        if value.endswith("x") and len(value) > 1:
            value = value[:-1]
        return all(
            self._mapping[self._offset + (position >> 3)]
            & (1 << (position & 7))
            for position in bloom_positions(
                value, self.bit_count, self.hash_count
            )
        )

    def close(self) -> None:
        self._mapping.close()
        self._handle.close()
