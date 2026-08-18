"""
B.0.1 -- snapshot data/raw/ to an immutable, hashed frozen version.

Run before any Stage B-E measurement. The dataset hash it prints goes into
`project_state.md`; MASTER_CONTEXT section 10 item 4 requires every session
to confirm that hash before measuring.

    python scripts/freeze_dataset.py                 # freeze a new version
    python scripts/freeze_dataset.py --verify        # re-hash the latest
    python scripts/freeze_dataset.py --note "..."    # annotate the freeze
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.freeze import (  # noqa: E402
    DEFAULT_FROZEN,
    DEFAULT_RAW,
    FrozenDatasetError,
    freeze,
    manifest_table,
    verify,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze or verify the dataset (B.0.1).")
    ap.add_argument("--verify", action="store_true", help="verify the latest frozen version")
    ap.add_argument("--version", type=int, default=None, help="version to verify")
    ap.add_argument("--note", default="", help="why this freeze was taken")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    args = ap.parse_args()

    if args.verify:
        try:
            m = verify(args.version, args.frozen)
        except FrozenDatasetError as e:
            print(f"!! {e}")
            return 1
        print(manifest_table(m))
        print("\nAll files match their manifest hashes.")
        return 0

    m = freeze(args.raw, args.frozen, note=args.note)
    print(manifest_table(m))
    print(
        f"\nFrozen to {args.frozen / ('v' + str(m['version']))}\n"
        f"\nRecord in project_state.md:\n"
        f"  frozen dataset v{m['version']}  dataset_sha256 {m['dataset_sha256']}\n"
        f"\nNote: data/ is gitignored, so the parquet files are local. The hash "
        f"above is the tracked, verifiable artifact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
