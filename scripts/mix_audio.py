from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal)) + 1e-12))


def match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        return audio[:target_len]
    repeats = int(np.ceil(target_len / max(len(audio), 1)))
    tiled = np.tile(audio, repeats)
    return tiled[:target_len]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_power = rms(clean)
    noise_power = rms(noise)
    target_noise_power = clean_power / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_power / max(noise_power, 1e-8))
    mixed = clean + scaled_noise
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32)
    return audio.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix clean speech and noise at target SNRs.")
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--noise_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(args.clean_dir.glob("*.wav"))
    noise_files = sorted(args.noise_dir.glob("*.wav"))
    if not clean_files or not noise_files:
        raise FileNotFoundError("Need at least one clean wav and one noise wav.")

    metadata_rows: list[dict[str, str | float]] = []

    for index, clean_path in enumerate(clean_files):
        clean = load_audio(clean_path, args.sample_rate)
        noise_path = noise_files[index % len(noise_files)]
        noise = load_audio(noise_path, args.sample_rate)
        noise = match_length(noise, len(clean))

        clean_out = args.output_dir / f"{clean_path.stem}_clean.wav"
        sf.write(clean_out, clean, args.sample_rate)

        for snr in args.snrs:
            mixed = mix_at_snr(clean, noise, snr)
            snr_tag = str(int(snr)) if float(snr).is_integer() else str(snr).replace(".", "p")
            noisy_out = args.output_dir / f"{clean_path.stem}_snr{snr_tag}.wav"
            sf.write(noisy_out, mixed, args.sample_rate)
            metadata_rows.append(
                {
                    "clean_path": str(clean_out),
                    "noisy_path": str(noisy_out),
                    "noise_path": str(noise_path),
                    "snr_db": snr,
                }
            )

    with (args.output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clean_path", "noisy_path", "noise_path", "snr_db"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Generated {len(metadata_rows)} noisy files in {args.output_dir}")


if __name__ == "__main__":
    main()
