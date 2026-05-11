"""Griffin-Lim phase refinement for magnitude-mask enhanced audio."""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from features.audio_features import compute_stft, invert_stft, load_audio


def griffin_lim(magnitude: np.ndarray, n_fft: int, hop_length: int,
                n_iter: int = 32) -> np.ndarray:
    """Iteratively refine phase to be consistent with the given magnitude.

    Args:
        magnitude: Target magnitude spectrogram (F, T), float32.
        n_fft, hop_length: STFT params (same as used for magnitude extraction).
        n_iter: Number of Griffin-Lim iterations. More = better convergence.

    Returns:
        Reconstructed time-domain signal, same length as original.
    """
    # Initialize with random phase
    angles = np.random.uniform(0, 2 * np.pi, magnitude.shape)
    # Add noisy phase as initialization bias for faster convergence
    # (pure random works but needs more iterations)

    for i in range(n_iter):
        # Build complex spectrogram, ISTFT back to time domain
        spec = magnitude * np.exp(1j * angles)
        audio = invert_stft(spec, n_fft, hop_length, -1)  # -1 = infer length

        # STFT back to freq domain, keep our target magnitude, update phase
        new_spec = compute_stft(audio, n_fft, hop_length)
        # Match length to target
        min_t = min(new_spec.shape[1], magnitude.shape[1])
        angles = np.angle(new_spec[:, :min_t])
        # Ensure angles has same shape as magnitude
        if angles.shape[1] < magnitude.shape[1]:
            pad = magnitude.shape[1] - angles.shape[1]
            angles = np.pad(angles, ((0, 0), (0, pad)), mode='edge')

    # Final ISTFT
    spec_final = magnitude * np.exp(1j * angles)
    return invert_stft(spec_final, n_fft, hop_length, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--enhanced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--n_iter", type=int, default=32,
                        help="Griffin-Lim iterations (default 32)")
    args = parser.parse_args()

    noisy = load_audio(str(args.noisy), 16000)
    enhanced = load_audio(str(args.enhanced), 16000)
    length = min(len(noisy), len(enhanced))
    noisy, enhanced = noisy[:length], enhanced[:length]

    # Get enhanced magnitude
    e_spec = compute_stft(enhanced, args.n_fft, args.hop_length)
    e_mag = np.abs(e_spec)

    # Griffin-Lim phase refinement
    refined = griffin_lim(e_mag, args.n_fft, args.hop_length, args.n_iter)
    refined = refined[:length].astype(np.float32)

    sf.write(str(args.output), refined, 16000)
    print(f"Griffin-Lim ({args.n_iter} iters): {args.enhanced} -> {args.output}")


if __name__ == "__main__":
    main()
