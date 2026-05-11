from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.audio_features import extract_feature_bundle, invert_stft, load_audio
from models.transformer_denoiser import TransformerDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained Transformer denoiser.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_wav", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    complex_mask = ckpt.get("complex_mask", False)
    causal = ckpt.get("causal", False)

    model = TransformerDenoiser(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        causal=causal,
        complex_mask=complex_mask,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    audio = load_audio(str(args.input_wav), ckpt["sample_rate"])
    features = extract_feature_bundle(
        audio=audio,
        sample_rate=ckpt["sample_rate"],
        n_fft=ckpt["n_fft"],
        hop_length=ckpt["hop_length"],
        n_mels=ckpt.get("n_mels", 40),
        rnnoise_bands=ckpt.get("rnnoise_bands", 22),
    )

    input_feature = features[ckpt.get("feature_type", "stft_log_mag")]
    mag = features["stft_mag"]
    phase = features["phase"]

    with torch.no_grad():
        mask = (
            model(torch.from_numpy(input_feature).unsqueeze(0)).squeeze(0).numpy()
        )

    if complex_mask:
        mask_real = mask[..., 0]
        mask_imag = mask[..., 1]
        noisy_real = features["stft_real"]
        noisy_imag = features["stft_imag"]
        enh_real = mask_real * noisy_real - mask_imag * noisy_imag
        enh_imag = mask_real * noisy_imag + mask_imag * noisy_real
        enhanced_spec = (enh_real + 1j * enh_imag).T
    else:
        enhanced_mag = (mask * mag).T
        enhanced_spec = enhanced_mag * np.exp(1j * phase.T)

    enhanced_audio = invert_stft(enhanced_spec, ckpt["n_fft"], ckpt["hop_length"], len(audio))

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, enhanced_audio, ckpt["sample_rate"])
    print(f"saved denoised audio to {args.output_wav}")


if __name__ == "__main__":
    main()
