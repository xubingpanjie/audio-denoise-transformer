from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.audio_features import extract_feature_bundle, load_audio
from models.transformer_denoiser import TransformerDenoiser


class PairedAudioDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        feature_type: str,
        n_mels: int,
        rnnoise_bands: int,
        complex_mask: bool = False,
    ) -> None:
        self.rows = list(csv.DictReader(metadata_csv.open("r", encoding="utf-8")))
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.rnnoise_bands = rnnoise_bands
        self.complex_mask = complex_mask

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        noisy = load_audio(row["noisy_path"], self.sample_rate)
        clean = load_audio(row["clean_path"], self.sample_rate)

        noisy_features = extract_feature_bundle(
            audio=noisy,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            rnnoise_bands=self.rnnoise_bands,
        )
        clean_features = extract_feature_bundle(
            audio=clean,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            rnnoise_bands=self.rnnoise_bands,
        )

        noisy_input = noisy_features[self.feature_type]
        noisy_mag = noisy_features["stft_mag"]

        result: dict[str, torch.Tensor] = {
            "noisy_input": torch.from_numpy(noisy_input),
            "noisy_mag": torch.from_numpy(noisy_mag),
        }

        if self.complex_mask:
            # cIRM target: C / S in complex domain
            noisy_real = noisy_features["stft_real"]
            noisy_imag = noisy_features["stft_imag"]
            clean_real = clean_features["stft_real"]
            clean_imag = clean_features["stft_imag"]
            denom = noisy_real**2 + noisy_imag**2 + 1e-8
            cirm_real = (clean_real * noisy_real + clean_imag * noisy_imag) / denom
            cirm_imag = (clean_imag * noisy_real - clean_real * noisy_imag) / denom
            cirm = np.stack([cirm_real, cirm_imag], axis=-1)  # (T, F, 2)
            result["target_mask"] = torch.from_numpy(cirm)
            result["noisy_real"] = torch.from_numpy(noisy_real)
            result["noisy_imag"] = torch.from_numpy(noisy_imag)
            result["clean_mag"] = torch.from_numpy(clean_features["stft_mag"])
        else:
            clean_mag = clean_features["stft_mag"]
            target_mask = np.clip(clean_mag / np.maximum(noisy_mag, 1e-6), 0.0, 1.0)
            result["target_mask"] = torch.from_numpy(target_mask)
            result["clean_mag"] = torch.from_numpy(clean_mag)

        return result


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["noisy_input"].shape[0] for item in batch)
    input_dim = batch[0]["noisy_input"].shape[1]
    output_dim = batch[0]["noisy_mag"].shape[1]

    noisy_input = torch.zeros(len(batch), max_len, input_dim)
    noisy_mag = torch.zeros(len(batch), max_len, output_dim)
    clean_mag = torch.zeros(len(batch), max_len, output_dim)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)

    has_complex = "noisy_real" in batch[0]
    if has_complex:
        target_mask = torch.zeros(len(batch), max_len, output_dim, 2)
        noisy_real = torch.zeros(len(batch), max_len, output_dim)
        noisy_imag = torch.zeros(len(batch), max_len, output_dim)
    else:
        target_mask = torch.zeros(len(batch), max_len, output_dim)

    for idx, item in enumerate(batch):
        frames = item["noisy_input"].shape[0]
        noisy_input[idx, :frames] = item["noisy_input"]
        noisy_mag[idx, :frames] = item["noisy_mag"]
        clean_mag[idx, :frames] = item["clean_mag"]
        target_mask[idx, :frames] = item["target_mask"]
        padding_mask[idx, :frames] = False
        if has_complex:
            noisy_real[idx, :frames] = item["noisy_real"]
            noisy_imag[idx, :frames] = item["noisy_imag"]

    result: dict[str, torch.Tensor] = {
        "noisy_input": noisy_input,
        "noisy_mag": noisy_mag,
        "clean_mag": clean_mag,
        "target_mask": target_mask,
        "padding_mask": padding_mask,
    }
    if has_complex:
        result["noisy_real"] = noisy_real
        result["noisy_imag"] = noisy_imag

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer spectral-mask denoiser.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature_type", type=str, default="stft_log_mag",
                        choices=["stft_log_mag", "log_mel", "rnnoise_like"])
    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--rnnoise_bands", type=int, default=22)
    parser.add_argument("--causal", action="store_true", help="Use causal attention for streaming support")
    parser.add_argument("--complex_mask", action="store_true", help="Predict complex IRM instead of magnitude mask")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint, continuing from the last saved epoch")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/transformer_best.pt"))
    parser.add_argument("--history_csv", type=Path, default=Path("outputs/metrics/transformer_loss_history.csv"))
    return parser.parse_args()


def apply_complex_mask(
    cirm: torch.Tensor, noisy_real: torch.Tensor, noisy_imag: torch.Tensor
) -> torch.Tensor:
    """Apply complex IRM to noisy STFT, return enhanced magnitude."""
    mask_real = cirm[..., 0]
    mask_imag = cirm[..., 1]
    enh_real = mask_real * noisy_real - mask_imag * noisy_imag
    enh_imag = mask_real * noisy_imag + mask_imag * noisy_real
    return torch.sqrt(enh_real**2 + enh_imag**2 + 1e-8)


def main() -> None:
    args = parse_args()
    metadata_csv = args.data_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_csv}")

    dataset = PairedAudioDataset(
        metadata_csv=metadata_csv,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        feature_type=args.feature_type,
        n_mels=args.n_mels,
        rnnoise_bands=args.rnnoise_bands,
        complex_mask=args.complex_mask,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    sample = dataset[0]
    model = TransformerDenoiser(
        input_dim=sample["noisy_input"].shape[1],
        output_dim=sample["noisy_mag"].shape[1],
        causal=args.causal,
        complex_mask=args.complex_mask,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss(reduction="none")

    # Frequency-weighted loss: emphasize lower freqs where speech energy concentrates
    freq_dim = sample["noisy_mag"].shape[1]
    freq_idx = torch.arange(freq_dim, dtype=torch.float32)
    freq_weight = 1.0 + 2.0 * (1.0 - freq_idx / freq_dim)  # 3x at DC, 1x at Nyquist
    freq_weight = freq_weight.to(device)

    start_epoch = 1
    best_loss = float("inf")
    history_rows: list[dict[str, float | int | str]] = []

    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for resume: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        history_rows = ckpt.get("history_rows", [])
        print(f"Resumed from epoch {start_epoch - 1} (best_loss={best_loss:.6f}), continuing at epoch {start_epoch}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)

    mode_tag = []
    if args.causal:
        mode_tag.append("causal")
    if args.complex_mask:
        mode_tag.append("cirm")
    tag = f"{args.feature_type}" + ("_" + "_".join(mode_tag) if mode_tag else "")

    print(f"Training {len(dataset)} samples, {len(loader)} batches/epoch, device={device}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        frame_count = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            noisy_input = batch["noisy_input"].to(device)
            noisy_mag = batch["noisy_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            pred_mask = model(noisy_input, src_key_padding_mask=padding_mask)

            if args.complex_mask:
                noisy_real = batch["noisy_real"].to(device)
                noisy_imag = batch["noisy_imag"].to(device)
                pred_clean_mag = apply_complex_mask(pred_mask, noisy_real, noisy_imag)
            else:
                pred_clean_mag = pred_mask * noisy_mag

            valid = (~padding_mask).unsqueeze(-1)
            loss = criterion(pred_clean_mag, clean_mag)
            loss = (loss * freq_weight * valid).sum() / valid.sum().clamp_min(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * noisy_mag.size(0)
            frame_count += noisy_mag.size(0)

            if batch_idx % 50 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_idx}/{len(loader)} loss={loss.item():.4f} elapsed={elapsed:.0f}s", flush=True)

        epoch_loss = running_loss / max(frame_count, 1)
        elapsed = time.time() - t0
        scheduler.step()
        history_rows.append({"epoch": epoch, "loss": epoch_loss, "feature_type": tag,
                             "lr": scheduler.get_last_lr()[0]})
        print(f"epoch={epoch} loss={epoch_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.0f}s", flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "history_rows": history_rows,
                "input_dim": sample["noisy_input"].shape[1],
                "output_dim": sample["noisy_mag"].shape[1],
                "sample_rate": args.sample_rate,
                "n_fft": args.n_fft,
                "hop_length": args.hop_length,
                "feature_type": args.feature_type,
                "n_mels": args.n_mels,
                "rnnoise_bands": args.rnnoise_bands,
                "causal": args.causal,
                "complex_mask": args.complex_mask,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint to {args.checkpoint}")

    with args.history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr", "feature_type"])
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"saved training history to {args.history_csv}")


if __name__ == "__main__":
    main()
