from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_loss_history(path: Path) -> tuple[list[int], list[float], str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    label = rows[0]["feature_type"] if rows else path.stem
    return epochs, losses, label


def read_metric_average(path: Path) -> dict[str, float | str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    average_row = rows[-1]
    return {
        "label": path.stem.replace("_metrics", ""),
        "pesq": float(average_row["pesq"]) if average_row["pesq"] else np.nan,
        "stoi": float(average_row["stoi"]) if average_row["stoi"] else np.nan,
        "sdr": float(average_row["sdr"]) if average_row["sdr"] else np.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves and metric bar charts.")
    parser.add_argument("--history_csvs", type=Path, nargs="*", default=[])
    parser.add_argument("--metric_csvs", type=Path, nargs="*", default=[])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/plots"))
    return parser.parse_args()


def plot_loss_curves(history_csvs: list[Path], output_dir: Path) -> None:
    if not history_csvs:
        return
    plt.figure(figsize=(8, 5))
    for path in history_csvs:
        epochs, losses, label = read_loss_history(path)
        plt.plot(epochs, losses, marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def plot_metric_bars(metric_csvs: list[Path], output_dir: Path) -> None:
    if not metric_csvs:
        return
    metric_rows = [read_metric_average(path) for path in metric_csvs]
    labels = [row["label"] for row in metric_rows]
    metrics = ["pesq", "stoi", "sdr"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric_name in zip(axes, metrics):
        values = [row[metric_name] for row in metric_rows]
        ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"][: len(labels)])
        ax.set_title("SDR" if metric_name == "sdr" else metric_name.upper())
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "metrics_bar.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(args.history_csvs, args.output_dir)
    plot_metric_bars(args.metric_csvs, args.output_dir)


if __name__ == "__main__":
    main()
