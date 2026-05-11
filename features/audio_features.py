from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.fft import rfftfreq
from scipy.signal import istft, resample_poly, stft


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32)
    return audio.astype(np.float32)


def compute_stft(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    _, _, spec = stft(
        audio,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )
    return spec


def invert_stft(spec: np.ndarray, n_fft: int, hop_length: int, target_length: int) -> np.ndarray:
    _, audio = istft(
        spec,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
        boundary=True,
    )
    return audio[:target_length].astype(np.float32)


def magnitude_phase(audio: np.ndarray, n_fft: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    spec = compute_stft(audio, n_fft, hop_length)
    return np.abs(spec).T.astype(np.float32), np.angle(spec).T.astype(np.float32)


def hz_to_bark(freq_hz: np.ndarray) -> np.ndarray:
    return 6.0 * np.arcsinh(freq_hz / 600.0)


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    fft_freqs = rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2.0]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    filterbank = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)

    for idx in range(n_mels):
        left, center, right = hz_points[idx], hz_points[idx + 1], hz_points[idx + 2]
        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if center > left:
            filterbank[idx, left_mask] = (fft_freqs[left_mask] - left) / (center - left)
        if right > center:
            filterbank[idx, right_mask] = (right - fft_freqs[right_mask]) / (right - center)

    return filterbank


def build_rnnoise_like_bands(sample_rate: int, n_fft: int, band_count: int) -> np.ndarray:
    freqs = rfftfreq(n_fft, d=1.0 / sample_rate)
    bark = hz_to_bark(freqs)
    edges = np.linspace(bark.min(), bark.max(), band_count + 2)
    band_filters = []
    for band_idx in range(band_count):
        left, center, right = edges[band_idx], edges[band_idx + 1], edges[band_idx + 2]
        weights = np.zeros_like(bark, dtype=np.float32)
        left_mask = (bark >= left) & (bark <= center)
        right_mask = (bark >= center) & (bark <= right)
        if center > left:
            weights[left_mask] = (bark[left_mask] - left) / (center - left)
        if right > center:
            weights[right_mask] = (right - bark[right_mask]) / (right - center)
        if weights.sum() > 0:
            weights /= weights.sum()
        band_filters.append(weights)
    return np.stack(band_filters, axis=0).astype(np.float32)


def extract_feature_bundle(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    rnnoise_bands: int,
) -> dict[str, np.ndarray]:
    spec = compute_stft(audio, n_fft, hop_length)
    magnitude = np.abs(spec).T.astype(np.float32)
    power = magnitude**2

    stft_log_mag = np.log1p(magnitude).astype(np.float32)

    mel_filter = build_mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
    log_mel = np.log1p(power @ mel_filter.T).astype(np.float32)

    rnnoise_filter = build_rnnoise_like_bands(sample_rate=sample_rate, n_fft=n_fft, band_count=rnnoise_bands)
    rnnoise_band_energy = np.log1p(power @ rnnoise_filter.T).astype(np.float32)
    rnnoise_delta = np.diff(rnnoise_band_energy, axis=0, prepend=rnnoise_band_energy[:1])
    rnnoise_like = np.concatenate([rnnoise_band_energy, rnnoise_delta], axis=1).astype(np.float32)

    return {
        "stft_log_mag": stft_log_mag,
        "log_mel": log_mel,
        "rnnoise_like": rnnoise_like,
        "stft_mag": magnitude,
        "phase": np.angle(spec).T.astype(np.float32),
        "stft_real": spec.real.T.astype(np.float32),
        "stft_imag": spec.imag.T.astype(np.float32),
    }
