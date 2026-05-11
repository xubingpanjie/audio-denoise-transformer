"""Batch inference for multiple checkpoints on test set."""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import csv

CKPTS = [
    ("STFT_baseline", PROJECT_ROOT / "checkpoints/transformer_stft_baseline.pt"),
    ("STFT_cIRM", PROJECT_ROOT / "checkpoints/transformer_stft_cirm.pt"),
    ("Mel_cIRM", PROJECT_ROOT / "checkpoints/transformer_mel_cirm.pt"),
    ("RNNoise_cIRM", PROJECT_ROOT / "checkpoints/transformer_rnlike_cirm.pt"),
    ("STFT_cIRM_causal", PROJECT_ROOT / "checkpoints/transformer_stft_cirm_causal.pt"),
]

METADATA = PROJECT_ROOT / "data/timit_mixed/metadata_test.csv"
OUT_BASE = PROJECT_ROOT / "outputs/transformer"


def main():
    with METADATA.open("r") as f:
        rows = list(csv.DictReader(f))
    print(f"Test set: {len(rows)} files, {len(CKPTS)} models")

    for name, ckpt in CKPTS:
        if not ckpt.exists():
            print(f"  SKIP {name}: checkpoint missing")
            continue

        out_dir = OUT_BASE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        done, skip = 0, 0

        for row in rows:
            noisy_path = (PROJECT_ROOT / row["noisy_path"]).resolve()
            out_wav = out_dir / f"{Path(row['noisy_path']).stem}_enhanced.wav"

            if out_wav.exists():
                skip += 1
                done += 1
                continue

            result = subprocess.run(
                [
                    sys.executable, "-u",
                    str(PROJECT_ROOT / "scripts/infer_transformer.py"),
                    "--checkpoint", str(ckpt),
                    "--input_wav", str(noisy_path),
                    "--output_wav", str(out_wav),
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  ERROR {name} on {Path(row['noisy_path']).stem}: {result.stderr.strip()}")
                continue
            done += 1

        elapsed = time.time() - t0
        print(f"  {name}: {done}/{len(rows)} files ({elapsed:.0f}s), {skip} cached")

    print("All inference done.")


if __name__ == "__main__":
    main()
