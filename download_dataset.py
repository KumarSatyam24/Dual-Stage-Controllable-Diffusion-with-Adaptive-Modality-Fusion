#!/usr/bin/env python3
"""
Google Drive Dataset Downloader for RAGAF-Diffusion (Vast.ai)
-------------------------------------------------------------
Usage:
    python download_dataset.py --gdrive-url <YOUR_GDRIVE_SHARE_LINK>

    or edit GDRIVE_FILE_ID below and run:
    python download_dataset.py

Supports:
    - Single zip file  (e.g. sketchy.zip)
    - Shared folder    (uses gdown --folder)
    - File ID directly (--file-id option)
"""

import argparse
import os
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

# ─── CONFIGURE THESE ──────────────────────────────────────────────────────────
# Paste your Google Drive share link or file/folder ID here after uploading.
GDRIVE_SHARE_LINK = ""   # e.g. "https://drive.google.com/file/d/XXXX/view?usp=sharing"
GDRIVE_FOLDER_LINK = ""  # e.g. "https://drive.google.com/drive/folders/XXXX?usp=sharing"

# Where to extract the dataset on vast.ai
DATASET_ROOT = Path("/root/datasets/sketchy")
# ──────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
BLUE   = "\033[0;34m"
YELLOW = "\033[1;33m"
NC     = "\033[0m"

def log(msg, color=BLUE):   print(f"{color}{msg}{NC}")
def ok(msg):                print(f"{GREEN}✓ {msg}{NC}")
def err(msg):               print(f"{RED}✗ {msg}{NC}"); sys.exit(1)
def warn(msg):              print(f"{YELLOW}⚠ {msg}{NC}")


def extract_file_id(url: str) -> tuple[str, bool]:
    """Return (file_id, is_folder) from a Google Drive share URL."""
    if "folders" in url:
        fid = url.split("folders/")[1].split("?")[0].split("/")[0]
        return fid, True
    elif "/d/" in url:
        fid = url.split("/d/")[1].split("/")[0].split("?")[0]
        return fid, False
    else:
        # Assume bare ID was passed
        return url.strip(), False


def download(file_id: str, dest: Path, is_folder: bool):
    """Download from Google Drive using gdown."""
    dest.mkdir(parents=True, exist_ok=True)

    if is_folder:
        log(f"Downloading Google Drive folder → {dest}")
        cmd = [
            "gdown", "--folder",
            f"https://drive.google.com/drive/folders/{file_id}",
            "--output", str(dest),
            "--remaining-ok",
        ]
    else:
        output_path = dest / "sketchy_dataset.zip"
        log(f"Downloading Google Drive file → {output_path}")
        cmd = [
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "--output", str(output_path),
            "--fuzzy",
        ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        err("Download failed. Check your share link and make sure the file is set to 'Anyone with the link'.")

    ok("Download complete")
    return dest / "sketchy_dataset.zip" if not is_folder else None


def extract(archive: Path, dest: Path):
    """Extract zip or tar archive."""
    log(f"Extracting {archive.name} …")
    if archive.suffix == ".zip" or archive.suffixes[-2:] == [".tar", ".gz"]:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive, "r") as z:
                z.extractall(dest)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                t.extractall(dest)
        else:
            err(f"Unknown archive format: {archive}")
    ok(f"Extracted to {dest}")
    archive.unlink()
    ok("Removed archive to save space")


def fix_structure(root: Path):
    """
    Ensure the final layout is:
        root/
          sketch/tx_000000000000/{category}/*.png
          photo/tx_000000000000/{category}/*.jpg

    Handles common cases where gdown extracts into a nested subfolder.
    """
    log("Checking directory structure …")

    # If extracted into a single subdirectory, lift contents up
    children = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(children) == 1 and not (root / "sketch").exists():
        inner = children[0]
        log(f"  Lifting contents of '{inner.name}/' up one level")
        for item in inner.iterdir():
            item.rename(root / item.name)
        inner.rmdir()

    has_sketch = (root / "sketch").exists()
    has_photo  = (root / "photo").exists()

    if has_sketch and has_photo:
        ok("Structure looks correct: sketch/ and photo/ both present")
    elif has_sketch and not has_photo:
        warn("photo/ folder missing — only sketches found. Upload the photo split too.")
    elif not has_sketch and not has_photo:
        warn("Neither sketch/ nor photo/ found. Check your archive structure.")
        log("Current contents of dataset root:")
        for p in root.iterdir():
            print(f"  {p.name}/")
    

def verify(root: Path):
    """Quick sanity check."""
    log("\nRunning quick verification …")
    sketch_dir = root / "sketch" / "tx_000000000000"
    photo_dir  = root / "photo"  / "tx_000000000000"

    if sketch_dir.exists():
        cats   = list(sketch_dir.iterdir())
        files  = list(sketch_dir.rglob("*.png"))
        ok(f"Sketches: {len(cats)} categories, {len(files)} files")
    else:
        warn("sketch/tx_000000000000/ not found")

    if photo_dir.exists():
        cats  = list(photo_dir.iterdir())
        files = list(photo_dir.rglob("*.jpg"))
        ok(f"Photos  : {len(cats)} categories, {len(files)} files")
    else:
        warn("photo/tx_000000000000/ not found")

    print()
    log("Dataset path is ready to use:", GREEN)
    print(f"  {root}")
    print()
    log("Next step — run the project verifier:", GREEN)
    print("  python verify_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Download Sketchy dataset from Google Drive")
    parser.add_argument("--gdrive-url",  default="", help="Google Drive share link (file or folder)")
    parser.add_argument("--file-id",     default="", help="Raw Google Drive file/folder ID")
    parser.add_argument("--output-dir",  default=str(DATASET_ROOT), help="Where to save the dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only fix structure")
    args = parser.parse_args()

    dest = Path(args.output_dir)

    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}RAGAF-Diffusion — Google Drive Dataset Downloader{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")

    if args.skip_download:
        fix_structure(dest)
        verify(dest)
        return

    # Resolve the Google Drive URL / ID
    url = args.gdrive_url or GDRIVE_SHARE_LINK or GDRIVE_FOLDER_LINK
    fid = args.file_id

    if not url and not fid:
        err(
            "No Google Drive link provided.\n\n"
            "  Option A: Pass it on the command line:\n"
            "    python download_dataset.py --gdrive-url 'https://drive.google.com/...'\n\n"
            "  Option B: Edit GDRIVE_SHARE_LINK at the top of this script."
        )

    if fid:
        file_id, is_folder = fid, ("folder" in fid)
    else:
        file_id, is_folder = extract_file_id(url)

    log(f"File ID : {file_id}")
    log(f"Type    : {'folder' if is_folder else 'file'}")
    log(f"Dest    : {dest}\n")

    archive = download(file_id, dest, is_folder)

    if archive and archive.exists():
        extract(archive, dest)

    fix_structure(dest)
    verify(dest)


if __name__ == "__main__":
    main()
