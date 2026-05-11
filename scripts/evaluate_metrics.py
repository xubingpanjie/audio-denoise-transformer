from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    from pesq import pesq
except ImportError:  # pragma: no cover
    pesq = None

try:
    from pystoi import stoi
except ImportError:  # pragma: no cover
    stoi = None


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32)
    return audio


def align_signals(clean: np.ndarray, enhanced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(clean), len(enhanced))
    return clean[:length], enhanced[:length]


def compute_sdr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    residual = clean - enhanced
    signal_power = float(np.sum(clean**2))
    residual_power = float(np.sum(residual**2) + 1e-12)
    return 10.0 * math.log10((signal_power + 1e-12) / residual_power)


def compute_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Alias kept for backward compatibility."""
    return compute_sdr(clean, enhanced)


def pair_paths(metadata_csv: Path, enhanced_dir: Path, mode: str) -> list[tuple[Path, Path, str]]:
    with metadata_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pairs: list[tuple[Path, Path, str]] = []
    project_root = metadata_csv.parents[2]
    for row in rows:
        clean_path = (project_root / row["clean_path"]).resolve()
        noisy_stem = Path(row["noisy_path"]).stem
        if mode == "rnnoise":
            enhanced_name = f"{noisy_stem}_rnnoise.wav"
        elif mode == "same_stem":
            enhanced_name = f"{noisy_stem}.wav"
        else:
            enhanced_name = mode.format(noisy_stem=noisy_stem)
        enhanced_path = (enhanced_dir / enhanced_name).resolve()
        pairs.append((clean_path, enhanced_path, noisy_stem))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate PESQ, STOI, and SDR for enhanced wav files.")
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--enhanced_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="rnnoise",
        help="rnnoise, same_stem, or a Python format string like {noisy_stem}_denoised.wav",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pair_paths(args.metadata_csv, args.enhanced_dir, args.pair_mode)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str | float]] = []
    if pesq is None:
        print("warning: PESQ package is unavailable in this Python environment; PESQ columns will be blank.")
    if stoi is None:
        print("warning: STOI package is unavailable in this Python environment; STOI columns will be blank.")

    for clean_path, enhanced_path, sample_id in pairs:
        if not enhanced_path.exists():
            print(f"skip missing enhanced file: {enhanced_path}")
            continue

        clean = load_audio(clean_path, args.sample_rate)
        enhanced = load_audio(enhanced_path, args.sample_rate)
        clean, enhanced = align_signals(clean, enhanced)

        result = {
            "sample_id": sample_id,
            "clean_path": str(clean_path),
            "enhanced_path": str(enhanced_path),
            "pesq": pesq(args.sample_rate, clean, enhanced, "wb") if pesq is not None else "",
            "stoi": stoi(clean, enhanced, args.sample_rate, extended=False) if stoi is not None else "",
            "sdr": compute_sdr(clean, enhanced),
        }
        results.append(result)
        pesq_text = f"{result['pesq']:.4f}" if result["pesq"] != "" else "N/A"
        stoi_text = f"{result['stoi']:.4f}" if result["stoi"] != "" else "N/A"
        print(f"{sample_id}: PESQ={pesq_text} STOI={stoi_text} SDR={result['sdr']:.4f}")

    if not results:
        raise RuntimeError("No evaluation results were produced.")

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "clean_path", "enhanced_path", "pesq", "stoi", "sdr"],
        )
        writer.writeheader()
        writer.writerows(results)

        numeric_pesq = [float(row["pesq"]) for row in results if row["pesq"] != ""]
        numeric_stoi = [float(row["stoi"]) for row in results if row["stoi"] != ""]
        averages = {
            "sample_id": "average",
            "clean_path": "",
            "enhanced_path": "",
            "pesq": float(np.mean(numeric_pesq)) if numeric_pesq else "",
            "stoi": float(np.mean(numeric_stoi)) if numeric_stoi else "",
            "sdr": float(np.mean([row["sdr"] for row in results])),
        }
        writer.writerow(averages)

    print(f"saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
