"""Wiener post-filter to refine magnitude mask enhanced audio."""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from features.audio_features import compute_stft, invert_stft, load_audio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--enhanced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Oversubtraction factor (higher = more aggressive denoising)")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="Spectral floor (prevents musical noise)")
    return parser.parse_args()


def main():
    args = parse_args()
    noisy = load_audio(str(args.noisy), 16000)
    enhanced = load_audio(str(args.enhanced), 16000)
    length = min(len(noisy), len(enhanced))
    noisy, enhanced = noisy[:length], enhanced[:length]

    # STFT
    n_spec = compute_stft(noisy, args.n_fft, args.hop_length)  # (F, T)
    e_spec = compute_stft(enhanced, args.n_fft, args.hop_length)

    n_mag = np.abs(n_spec)
    n_phase = np.angle(n_spec)
    e_mag = np.abs(e_spec)

    # Estimate residual noise power from the difference
    residual_spec = n_spec - e_spec
    noise_power = np.abs(residual_spec) ** 2

    # Time-smooth noise estimate
    alpha_n = 0.98
    noise_power_smooth = np.zeros_like(noise_power)
    noise_power_smooth[:, 0] = noise_power[:, 0]
    for t in range(1, noise_power.shape[1]):
        noise_power_smooth[:, t] = alpha_n * noise_power_smooth[:, t - 1] + (1 - alpha_n) * noise_power[:, t]

    # Wiener gain
    signal_power = np.maximum(e_mag ** 2 - args.alpha * noise_power_smooth, 0.0)
    gain = signal_power / (signal_power + noise_power_smooth + 1e-8)

    # Spectral floor
    gain = np.maximum(gain, args.beta)

    # Apply gain
    refined_mag = gain * n_mag
    refined_spec = refined_mag * np.exp(1j * n_phase)
    refined_audio = invert_stft(refined_spec, args.n_fft, args.hop_length, length)

    sf.write(str(args.output), refined_audio, 16000)
    print(f"Wiener post-filter: {args.enhanced} -> {args.output}")


if __name__ == "__main__":
    main()
