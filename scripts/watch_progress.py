"""Real-time training progress dashboard. Ctrl+C to exit."""
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

TASKS = [
    {"name": "STFT+cIRM        ", "log": "training_stft_cirm.log", "total": 30},
    {"name": "Mel+cIRM         ", "log": "training_mel_cirm.log", "total": 30},
    {"name": "RNNoise+cIRM     ", "log": "training_rnlike_cirm.log", "total": 30},
]


def count_epochs(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    try:
        text = log_path.read_text()
        return text.count("epoch=")
    except Exception:
        return 0


def bar(done: int, total: int, width: int = 20) -> str:
    n = min(int(done / total * width), width)
    return "#" * n + "-" * (width - n)


def main():
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print("=" * 55)
            print("  训 练 进 度 监 控  (Ctrl+C 退出)")
            print("=" * 55)
            print()

            for t in TASKS:
                done = count_epochs(OUTPUTS / t["log"])
                pct = int(done / t["total"] * 100) if t["total"] else 0
                b = bar(done, t["total"])
                last_loss = ""
                log_path = OUTPUTS / t["log"]
                if log_path.exists():
                    try:
                        lines = log_path.read_text().strip().split("\n")
                        for line in reversed(lines):
                            if "epoch=" in line:
                                parts = line.split()
                                for p in parts:
                                    if p.startswith("loss="):
                                        last_loss = f"  latest: {p}"
                                        break
                                break
                    except Exception:
                        pass
                print(f"  {b} {pct:>3}% [{done}/{t['total']}] {t['name']}{last_loss}")

            print()
            print("=" * 55)
            sys.stdout.flush()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n退出监控。")


if __name__ == "__main__":
    main()
