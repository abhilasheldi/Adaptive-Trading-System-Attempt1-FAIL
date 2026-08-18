"""
Dataset freezing -- BUILD_PLAN_v2 B.0.1.

=========================================================================
WHY
=========================================================================
The BTC 1h row count drifted 78,378 -> 78,381 -> 78,226 -> 78,666 across
stages. Every one of those numbers appears in a recorded measurement. A.6's
corrected ceiling table was computed against 78,226 labelled rows; that
number cannot be reproduced against `data/raw/` today, because `data/raw/`
has moved on. Cross-stage comparison against a drifting dataset is not
comparison.

The drift is not a bug -- an incremental ingest appending new bars is
correct behaviour. The bug is measuring against a moving target and
reporting the result as if it were stable.

=========================================================================
CONTRACT
=========================================================================
- `freeze()` snapshots `data/raw/` to `data/frozen/v<N>/` and writes a
  manifest with a SHA-256 per file plus row count and window.
- `load_frozen()` is what Stage B-E measurement code calls. It verifies the
  hash on every read and raises on mismatch.
- Re-pulls are deliberate: a new freeze gets a NEW version directory. An
  existing version is never overwritten, so a hash recorded in
  `project_state.md` keeps meaning what it meant.

`data/` is gitignored, so the frozen copy is local. The manifest hash
recorded in `project_state.md` is what makes it verifiable across machines
-- it is the tracked artifact, not the parquet.

Hashing is over the PARQUET BYTES, not the DataFrame. Parquet writes are
not bit-reproducible across library versions, so a byte hash answers "is
this the same file I froze" -- which is the question -- rather than "does
this contain the same data", which would need a canonicalisation step whose
own version drift would reintroduce the problem it solved.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_RAW = Path("data/raw")
DEFAULT_FROZEN = Path("data/frozen")
MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


class FrozenDatasetError(RuntimeError):
    """Raised when a frozen file fails verification, or a version is misused."""


@dataclass(frozen=True)
class FrozenFile:
    filename: str
    sha256: str
    bytes: int
    rows: int
    start: Optional[str]
    end: Optional[str]
    columns: List[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _describe(path: Path) -> FrozenFile:
    df = pd.read_parquet(path)
    has_dt = isinstance(df.index, pd.DatetimeIndex) and len(df)
    return FrozenFile(
        filename=path.name,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        rows=len(df),
        start=str(df.index.min()) if has_dt else None,
        end=str(df.index.max()) if has_dt else None,
        columns=[str(c) for c in df.columns],
    )


def next_version(frozen_root: Path = DEFAULT_FROZEN) -> int:
    """Next unused version number. Existing versions are never overwritten."""
    if not frozen_root.exists():
        return 1
    used = [
        int(p.name[1:])
        for p in frozen_root.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    return max(used, default=0) + 1


def freeze(
    raw_dir: Path = DEFAULT_RAW,
    frozen_root: Path = DEFAULT_FROZEN,
    version: Optional[int] = None,
    note: str = "",
) -> dict:
    """Snapshot every parquet in `raw_dir` into a new frozen version.

    Refuses to overwrite an existing version -- that is the whole guarantee.
    """
    raw_dir, frozen_root = Path(raw_dir), Path(frozen_root)
    if not raw_dir.exists():
        raise FrozenDatasetError(f"{raw_dir} does not exist -- nothing to freeze.")

    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FrozenDatasetError(f"No parquet files in {raw_dir}.")

    version = version if version is not None else next_version(frozen_root)
    target = frozen_root / f"v{version}"
    if target.exists():
        raise FrozenDatasetError(
            f"{target} already exists. Frozen versions are immutable -- a re-pull "
            f"takes the next version number so that a hash recorded in "
            f"project_state.md keeps meaning what it meant. Pass version=None."
        )
    target.mkdir(parents=True)

    described: List[FrozenFile] = []
    for src in files:
        shutil.copy2(src, target / src.name)
        described.append(_describe(target / src.name))

    manifest = {
        "version": version,
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_dir": str(raw_dir),
        "note": note,
        "files": [asdict(f) for f in described],
        # Hash of the per-file hashes: one number identifying the whole
        # snapshot, which is what project_state.md records.
        "dataset_sha256": _dataset_hash(described),
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _dataset_hash(files: List[FrozenFile]) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.filename):
        h.update(f.filename.encode())
        h.update(f.sha256.encode())
    return h.hexdigest()


def latest_version(frozen_root: Path = DEFAULT_FROZEN) -> int:
    v = next_version(frozen_root) - 1
    if v < 1:
        raise FrozenDatasetError(
            f"No frozen dataset under {frozen_root}. Run scripts/freeze_dataset.py "
            f"before any Stage B-E measurement (MASTER_CONTEXT section 10, item 4)."
        )
    return v


def read_manifest(version: Optional[int] = None, frozen_root: Path = DEFAULT_FROZEN) -> dict:
    frozen_root = Path(frozen_root)
    version = version if version is not None else latest_version(frozen_root)
    path = frozen_root / f"v{version}" / MANIFEST_NAME
    if not path.exists():
        raise FrozenDatasetError(f"No manifest at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def verify(version: Optional[int] = None, frozen_root: Path = DEFAULT_FROZEN) -> dict:
    """Re-hash every file in a frozen version against its manifest.

    Returns the manifest on success; raises listing every mismatch on
    failure. Checks all files before raising rather than stopping at the
    first, because "which files moved" is the useful question.
    """
    frozen_root = Path(frozen_root)
    manifest = read_manifest(version, frozen_root)
    root = frozen_root / f"v{manifest['version']}"

    problems: List[str] = []
    for entry in manifest["files"]:
        path = root / entry["filename"]
        if not path.exists():
            problems.append(f"  MISSING  {entry['filename']}")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(
                f"  CHANGED  {entry['filename']}\n"
                f"           manifest {entry['sha256'][:16]}...  actual {actual[:16]}..."
            )
    if problems:
        raise FrozenDatasetError(
            f"Frozen dataset v{manifest['version']} failed verification:\n"
            + "\n".join(problems)
            + "\n\nA frozen version is immutable by contract. Re-freeze to a new "
            "version rather than repairing this one."
        )
    return manifest


def load_frozen(
    filename: str,
    version: Optional[int] = None,
    frozen_root: Path = DEFAULT_FROZEN,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """Read one file from the frozen dataset, verifying its hash by default.

    This is the entry point Stage B-E measurement code uses instead of
    reading `data/raw/` directly.
    """
    frozen_root = Path(frozen_root)
    manifest = read_manifest(version, frozen_root)
    root = frozen_root / f"v{manifest['version']}"
    path = root / filename

    if not path.exists():
        available = ", ".join(sorted(e["filename"] for e in manifest["files"]))
        raise FrozenDatasetError(f"{filename} not in v{manifest['version']}. Have: {available}")

    if verify_hash:
        entry = next(e for e in manifest["files"] if e["filename"] == filename)
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise FrozenDatasetError(
                f"{filename} in v{manifest['version']} does not match its manifest hash "
                f"(manifest {entry['sha256'][:16]}..., actual {actual[:16]}...). "
                f"Stop and reconcile -- a measurement against this file is not "
                f"comparable to anything else recorded under this version."
            )
    return pd.read_parquet(path)


def manifest_table(manifest: dict) -> str:
    """Human-readable manifest, for project_state.md and ingest reports."""
    lines = [
        f"Frozen dataset v{manifest['version']}  ({manifest['frozen_at']})",
        f"dataset_sha256: {manifest['dataset_sha256']}",
        "",
        f"{'file':<34}{'rows':>10}  {'sha256[:16]':<18}{'window'}",
        "-" * 100,
    ]
    for e in sorted(manifest["files"], key=lambda x: x["filename"]):
        window = f"{(e['start'] or '')[:10]} -> {(e['end'] or '')[:10]}" if e["start"] else ""
        lines.append(f"{e['filename']:<34}{e['rows']:>10,}  {e['sha256'][:16]:<18}{window}")
    return "\n".join(lines)


__all__ = [
    "FrozenDatasetError",
    "FrozenFile",
    "freeze",
    "verify",
    "load_frozen",
    "read_manifest",
    "latest_version",
    "next_version",
    "sha256_file",
    "manifest_table",
]
