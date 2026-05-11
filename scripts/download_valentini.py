"""
Download the Valentini et al. (2017) dataset with resume + retry support.

VCTK clean speech mixed with DEMAND noise at 4 SNRs (0, 5, 10, 15 dB).
28 training speakers, 2 test speakers. ~11 GB zipped.

Reference: https://datashare.ed.ac.uk/handle/10283/2791
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests

URLS = {
    "noisy_trainset_28spk_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip",
    "clean_trainset_28spk_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
    "noisy_testset_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
    "clean_testset_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
}

CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds base, exponential backoff


def download_file(url: str, dest: Path) -> None:
    part_path = dest.with_suffix(dest.suffix + ".part")
    resume_pos = part_path.stat().st_size if part_path.exists() else 0

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    for attempt in range(1, MAX_RETRIES + 1):
        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
            print(f"  Resuming from {resume_pos / (1024*1024):.0f} MB ...")

        try:
            response = session.get(url, stream=True, timeout=(30, 300), headers=headers)
            total = int(response.headers.get("content-length", 0)) + resume_pos

            if total <= resume_pos:
                print("  Already complete.")
                break

            with open(part_path, "ab") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        break
                    f.write(chunk)
                    resume_pos += len(chunk)
                    if total:
                        pct = resume_pos / total * 100
                        mb = resume_pos / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"\r    {mb:.0f}/{mb_total:.0f} MB ({pct:.0f}%)", end="", flush=True)

            print()
            part_path.rename(dest)
            return  # success

        except (requests.ConnectionError, requests.ChunkedEncodingError,
                ConnectionResetError, TimeoutError, OSError) as e:
            part_path.touch()  # ensure part file exists for resume tracking
            resume_pos = part_path.stat().st_size
            wait = RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"\n  Retry {attempt}/{MAX_RETRIES} in {wait}s ({type(e).__name__})")
                time.sleep(wait)
            else:
                print(f"\n  Failed after {MAX_RETRIES} retries. Partial: {resume_pos/1024/1024:.0f} MB")
                raise


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Valentini VCTK+DEMAND dataset")
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory for the downloaded dataset",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=list(URLS),
        help="Skip specified archives",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        zip_path = args.output_root / f"{name}.zip"
        if zip_path.exists():
            print(f"[skip] {zip_path.name} already downloaded")
        else:
            print(f"[{name}]")
            download_file(url, zip_path)
        extract_zip(zip_path, args.output_root)

    print(f"\nDone. Dataset extracted to {args.output_root}")
    print("Expected layout:")
    print(f"  {args.output_root}/noisy_trainset_28spk_wav/")
    print(f"  {args.output_root}/clean_trainset_28spk_wav/")
    print(f"  {args.output_root}/noisy_testset_wav/")
    print(f"  {args.output_root}/clean_testset_wav/")


if __name__ == "__main__":
    main()
