"""SQLite-backed run archive for coalition evaluation history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..masking.codec import PackedMask
from ..pipeline.manifest import RunManifest


@dataclass(frozen=True, slots=True)
class CoalitionRecord:
    """One persisted coalition evaluation row."""

    record_id: int
    """Monotonic archive identifier for the coalition row."""

    snapshot_id: str
    """Conversation snapshot identifier evaluated for this coalition."""

    coalition_key: str
    """Stable deduplication key for the coalition evaluation."""

    mask: PackedMask
    """Packed coalition mask bytes and original bit length."""

    absence_policy: str
    """Registered absence-policy identifier used for rendering."""

    model_id: str
    """Backend model identifier used for generation."""

    generation_text: str
    """Generated model text for the coalition."""

    utility: float
    """Utility score assigned to the generated output."""

    elapsed_ms: float
    """Wall-clock generation time in milliseconds."""

    cache_hit: bool
    """Whether the generation was served from an in-memory cache."""


@dataclass(frozen=True, slots=True)
class CoalitionRecordDraft:
    """Input payload used when appending a coalition record."""

    snapshot_id: str
    """Conversation snapshot identifier evaluated for this coalition."""

    coalition_key: str
    """Stable deduplication key for the coalition evaluation."""

    mask: PackedMask
    """Packed coalition mask bytes and original bit length."""

    absence_policy: str
    """Registered absence-policy identifier used for rendering."""

    model_id: str
    """Backend model identifier used for generation."""

    generation_text: str
    """Generated model text for the coalition."""

    utility: float
    """Utility score assigned to the generated output."""

    elapsed_ms: float
    """Wall-clock generation time in milliseconds."""

    cache_hit: bool
    """Whether the generation was served from an in-memory cache."""


class RunArchive:
    """Persist coalition records to SQLite with generation text stored as blobs."""

    def __init__(
        self,
        root: Path,
        manifest: RunManifest,
        flush_every: int = 50,
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._flush_every = flush_every
        self._pending = 0
        self._blobs = root / "blobs"
        self._db_path = root / "archive.sqlite"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    @classmethod
    def open(
        cls,
        root: Path,
        manifest: RunManifest,
        flush_every: int = 50,
    ) -> RunArchive:
        """Create a run archive directory and open its SQLite database."""
        root.mkdir(parents=True, exist_ok=True)
        archive = cls(root, manifest, flush_every=flush_every)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return archive

    def append(self, draft: CoalitionRecordDraft) -> int:
        """Append one coalition record and return its archive identifier."""
        cursor = self._conn.execute(
            """
            INSERT INTO coalition_records (
                snapshot_id,
                coalition_key,
                mask_words,
                mask_n_bits,
                absence_policy,
                model_id,
                utility,
                elapsed_ms,
                cache_hit,
                generation_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft.snapshot_id,
                draft.coalition_key,
                draft.mask.words,
                draft.mask.n_bits,
                draft.absence_policy,
                draft.model_id,
                draft.utility,
                draft.elapsed_ms,
                int(draft.cache_hit),
                "",
            ),
        )
        if cursor.lastrowid is None:
            msg = "SQLite insert did not return a row id"
            raise RuntimeError(msg)
        record_id = int(cursor.lastrowid)
        blob_name = f"{record_id}.txt"
        (self._blobs / blob_name).write_text(draft.generation_text, encoding="utf-8")
        self._conn.execute(
            "UPDATE coalition_records SET generation_blob = ? WHERE record_id = ?",
            (blob_name, record_id),
        )
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()
        return record_id

    def flush(self) -> None:
        """Commit pending archive writes to disk."""
        self._conn.commit()
        self._pending = 0

    def history_lazy(self) -> Iterator[CoalitionRecord]:
        """Iterate coalition records one row at a time without bulk preloading."""
        cursor = self._conn.execute(
            """
            SELECT
                record_id,
                snapshot_id,
                coalition_key,
                mask_words,
                mask_n_bits,
                absence_policy,
                model_id,
                utility,
                elapsed_ms,
                cache_hit,
                generation_blob
            FROM coalition_records
            ORDER BY record_id
            """
        )
        for row in cursor:
            generation_text = (self._blobs / row["generation_blob"]).read_text(
                encoding="utf-8"
            )
            yield CoalitionRecord(
                record_id=int(row["record_id"]),
                snapshot_id=str(row["snapshot_id"]),
                coalition_key=str(row["coalition_key"]),
                mask=PackedMask(
                    words=bytes(row["mask_words"]),
                    n_bits=int(row["mask_n_bits"]),
                ),
                absence_policy=str(row["absence_policy"]),
                model_id=str(row["model_id"]),
                generation_text=generation_text,
                utility=float(row["utility"]),
                elapsed_ms=float(row["elapsed_ms"]),
                cache_hit=bool(row["cache_hit"]),
            )

    def close(self) -> None:
        """Flush and close the archive database connection."""
        self.flush()
        self._conn.close()

    def __enter__(self) -> RunArchive:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coalition_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                coalition_key TEXT NOT NULL,
                mask_words BLOB NOT NULL,
                mask_n_bits INTEGER NOT NULL,
                absence_policy TEXT NOT NULL,
                model_id TEXT NOT NULL,
                utility REAL NOT NULL,
                elapsed_ms REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                generation_blob TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
