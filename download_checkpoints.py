#!/usr/bin/env python3
"""Download checkpoints from HuggingFace with resume capability."""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, output_path, chunk_size=8192):
    """Download file with progress bar and resume capability."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and get its size
    resume_byte_pos = output_path.stat().st_size if output_path.exists() else 0

    # Set up headers for resume
    headers = {}
    if resume_byte_pos > 0:
        headers['Range'] = f'bytes={resume_byte_pos}-'
        print(f"Resuming download from byte {resume_byte_pos}")

    # Make request
    response = requests.get(url, headers=headers, stream=True, timeout=30)

    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    if 'content-range' in response.headers:
        total_size = int(response.headers['content-range'].split('/')[-1])

    # Determine mode
    mode = 'ab' if resume_byte_pos > 0 else 'wb'

    # Download with progress bar
    with open(output_path, mode) as f:
        with tqdm(total=total_size, initial=resume_byte_pos, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✓ Downloaded: {output_path}")

def main():
    base_dir = Path(__file__).parent / "checkpoints"

    # Stage 1 checkpoint
    stage1_files = [
        ("https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints/resolve/main/stage1_with_ssim/epoch_18.pt",
         base_dir / "stage1_with_ssim" / "epoch_18.pt")
    ]

    # Stage 2 checkpoints - available files from HuggingFace
    stage2_epochs = [2, 4, 6, 8, 10]
    stage2_files = [
        (f"https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints/resolve/main/stage2/epoch_{i}.pt",
         base_dir / "stage2" / f"epoch_{i}.pt")
        for i in stage2_epochs
    ]
    # Add final checkpoint
    stage2_files.append(
        ("https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints/resolve/main/stage2/final.pt",
         base_dir / "stage2" / "final.pt")
    )

    all_files = stage1_files + stage2_files

    print(f"Downloading {len(all_files)} checkpoint files...")
    print(f"Destination: {base_dir}\n")

    for url, output_path in all_files:
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"✗ Failed to download {output_path.name}: {e}")
            continue

    print("\nDownload complete!")

if __name__ == "__main__":
    main()
