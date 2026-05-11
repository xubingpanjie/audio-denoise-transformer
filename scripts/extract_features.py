from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.audio_features import extract_feature_bundle, load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract STFT, log-mel, and simplified RNNoise-style band features.")
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--rnnoise_bands", type=int, default=22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(Path().glob(args.input_glob))
    if not inputs:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    manifest: list[dict[str, str]] = []
    for wav_path in inputs:
        audio = load_audio(str(wav_path), args.sample_rate)
        features = extract_feature_bundle(
            audio=audio,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            rnnoise_bands=args.rnnoise_bands,
        )
        features = {key: value for key, value in features.items() if key in {"stft_log_mag", "log_mel", "rnnoise_like"}}
        out_path = args.output_dir / f"{wav_path.stem}_features.npz"
        np.savez_compressed(out_path, **features)
        manifest.append({"input_wav": str(wav_path), "feature_npz": str(out_path)})
        print(f"saved features: {out_path}")

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
