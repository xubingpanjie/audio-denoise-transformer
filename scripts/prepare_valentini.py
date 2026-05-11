"""
Prepare Valentini dataset for training.

The Valentini dataset has:
  - noisy_trainset_28spk_wav/  +  clean_trainset_28spk_wav/
  - noisy_testset_wav/          +  clean_testset_wav/

Noisy and clean files share the same basename. This script creates
metadata.csv files that the existing train_transformer.py can consume directly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_metadata(
    noisy_dir: Path,
    clean_dir: Path,
    output_csv: Path,
) -> None:
    noisy_files = sorted(noisy_dir.glob("*.wav"))
    rows: list[dict[str, str]] = []
    for noisy_path in noisy_files:
        clean_path = clean_dir / noisy_path.name
        if not clean_path.exists():
            print(f"  Warning: no clean match for {noisy_path.name}, skipping")
            continue
        rows.append({
            "clean_path": str(clean_path.resolve()),
            "noisy_path": str(noisy_path.resolve()),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clean_path", "noisy_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} pairs to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare metadata.csv from Valentini dataset layout"
    )
    parser.add_argument(
        "--valentini_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "valentini",
        help="Root where the 4 valentini zip archives were extracted",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "valentini_prepared",
        help="Where to write the metadata.csv files",
    )
    args = parser.parse_args()

    root = args.valentini_root

    # ---- train ----
    print("[1/2] Train set")
    build_metadata(
        noisy_dir=root / "noisy_trainset_28spk_wav",
        clean_dir=root / "clean_trainset_28spk_wav",
        output_csv=args.output_root / "train" / "metadata.csv",
    )

    # ---- test ----
    print("[2/2] Test set")
    build_metadata(
        noisy_dir=root / "noisy_testset_wav",
        clean_dir=root / "clean_testset_wav",
        output_csv=args.output_root / "test" / "metadata.csv",
    )

    print("\nDone. Train with, e.g.:")
    print(f'  python scripts/train_transformer.py --data_dir {args.output_root / "train"} --epochs 50 --batch_size 8')
    print(f"  (Test set: {args.output_root / 'test' / 'metadata.csv'})")


if __name__ == "__main__":
    main()
