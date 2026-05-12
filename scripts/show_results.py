"""Print the final comparison table for all methods."""
import csv

methods = [
    ("noisy_baseline", "含噪不处理"),
    ("noisereduce_v2", "noisereduce"),
    ("rnnoise_v3", "RNNoise"),
    ("gtcrn_v3", "GTCRN"),
    ("STFT_improved", "Transformer"),
]

print(f'{"方法":<16} {"STOI":>8} {"SDR(dB)":>10}')
print("-" * 36)
for csv_name, label in methods:
    with open(f"outputs/metrics/{csv_name}_metrics.csv") as f:
        avg = [r for r in csv.reader(f) if r[0] == "average"][0]
    print(f"{label:<16} {float(avg[4]):>8.4f} {float(avg[5]):>10.2f}")
