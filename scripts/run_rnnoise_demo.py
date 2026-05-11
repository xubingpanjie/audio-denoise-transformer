from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    from pyrnnoise import RNNoise
except ImportError as exc:
    raise SystemExit("pyrnnoise is not installed. Run: python -m pip install pyrnnoise") from exc

RNNOISE_SR = 48000


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    # Use high-quality polyphase resampling
    return resample_poly(audio, target_sr, orig_sr, window=("kaiser", 10.0)).astype(np.float32)


def rnnoise_denoise_in_memory(denoiser: RNNoise, audio_48k: np.ndarray) -> np.ndarray:
    """Apply RNNoise fully in-memory, avoiding intermediate WAV file I/O.

    Uses denoise_chunk to bypass the double 16-bit quantization that
    denoise_wav introduces (write temp wav → read → process → write → read).
    """
    # RNNoise processes in 10ms frames at 48kHz → 480 samples
    frame_size = 480
    output_chunks: list[np.ndarray] = []

    for start in range(0, len(audio_48k), frame_size):
        chunk = audio_48k[start : start + frame_size]
        partial = (start + frame_size) >= len(audio_48k)
        # Pad if needed
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        for _, denoised_frame in denoiser.denoise_chunk(chunk, partial=partial):
            output_chunks.append(denoised_frame.copy())

    denoised = np.concatenate(output_chunks)
    return denoised[: len(audio_48k)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise wav files with RNNoise (in-memory pipeline).")
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_sample_rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    denoiser = RNNoise(sample_rate=RNNOISE_SR)

    inputs = sorted(Path().glob(args.input_glob))
    if not inputs:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    for wav_path in inputs:
        audio, orig_sr = sf.read(str(wav_path))
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Upsample 16k → 48k in memory
        audio_48k = resample(audio, orig_sr, RNNOISE_SR)

        # RNNoise in-memory (no temp files, no 16-bit quantization)
        denoised_48k = rnnoise_denoise_in_memory(denoiser, audio_48k)

        # Downsample 48k → 16k in memory
        enhanced = resample(denoised_48k, RNNOISE_SR, args.output_sample_rate)

        out_path = args.output_dir / f"{wav_path.stem}_rnnoise.wav"
        sf.write(str(out_path), enhanced, args.output_sample_rate)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
