from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split paired noisy-clean data into train/val/test folders.")
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    project_root = base_dir.parent.parent
    return (project_root / path).resolve()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def split_counts(total_items: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = max(1, int(total_items * train_ratio))
    val_count = max(1, int(total_items * val_ratio))
    test_count = total_items - train_count - val_count

    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    while train_count + val_count + test_count > total_items:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1

    return train_count, val_count, test_count


def copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    metadata_path = args.source_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 3:
        raise RuntimeError("Need at least 3 paired samples to create train/val/test splits.")

    random.Random(args.seed).shuffle(rows)
    train_count, val_count, test_count = split_counts(len(rows), args.train_ratio, args.val_ratio)

    splits = {
        "train": rows[:train_count],
        "val": rows[train_count : train_count + val_count],
        "test": rows[train_count + val_count : train_count + val_count + test_count],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in splits.items():
        split_dir = args.output_root / split_name
        clean_dir = split_dir / "clean"
        noisy_dir = split_dir / "noisy"
        noise_dir = split_dir / "noise"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
        noise_dir.mkdir(parents=True, exist_ok=True)

        output_rows: list[dict[str, str | float]] = []
        for row in split_rows:
            clean_src = resolve_path(args.source_dir, row["clean_path"])
            noisy_src = resolve_path(args.source_dir, row["noisy_path"])
            noise_src = resolve_path(args.source_dir, row["noise_path"])

            clean_dst = clean_dir / clean_src.name
            noisy_dst = noisy_dir / noisy_src.name
            noise_dst = noise_dir / noise_src.name

            copy_if_needed(clean_src, clean_dst)
            copy_if_needed(noisy_src, noisy_dst)
            copy_if_needed(noise_src, noise_dst)

            output_rows.append(
                {
                    "clean_path": str(clean_dst),
                    "noisy_path": str(noisy_dst),
                    "noise_path": str(noise_dst),
                    "snr_db": row["snr_db"],
                }
            )

        with (split_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["clean_path", "noisy_path", "noise_path", "snr_db"])
            writer.writeheader()
            writer.writerows(output_rows)

    print(
        "Created splits:",
        f"train={len(splits['train'])}",
        f"val={len(splits['val'])}",
        f"test={len(splits['test'])}",
    )


if __name__ == "__main__":
    main()
