#!/usr/bin/env python3
"""
Dataset Sync Progress Monitor
Run this to watch the Google Drive sync progress in real time.
Usage: python monitor_sync.py
"""

import time
import os
from pathlib import Path

SKETCHY_ROOT = Path("/workspace/sketchy")
PHOTO_DIR    = SKETCHY_ROOT / "photo" / "tx_000000000000"
SKETCH_DIR   = SKETCHY_ROOT / "sketch" / "tx_000000000000"
TARGET_CATS  = 125

def count(path: Path, ext: str) -> tuple[int, int]:
    """Returns (num_categories, num_files)."""
    if not path.exists():
        return 0, 0
    cats  = [d for d in path.iterdir() if d.is_dir()]
    files = list(path.rglob(f"*.{ext}"))
    return len(cats), len(files)

def bar(current, total, width=30) -> str:
    filled = int(width * current / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {current}/{total}"

def missing(path: Path, target_cats: list[str]) -> list[str]:
    if not path.exists():
        return target_cats
    existing = {d.name for d in path.iterdir() if d.is_dir()}
    return sorted(set(target_cats) - existing)

# Full 125 Sketchy categories
ALL_CATEGORIES = [
    "airplane","alarm_clock","ant","ape","apple","armor","axe","banana","bat","bear",
    "bee","beetle","bell","bench","bicycle","blimp","bread","butterfly","camel","candle",
    "cannon","car_(sedan)","castle","cat","chair","chicken","church","couch","cow","crab",
    "crocodilian","cup","deer","dog","dolphin","door","duck","elephant","eyeglasses",
    "fan","fantasy","fire_truck","fish","flower","flying_saucer","frog","giraffe",
    "gorilla","guitar","hamburger","hammer","harp","hat","hedgehog","helicopter",
    "hermit_crab","horse","hot-air_balloon","hot_dog","hourglass","jack-o-lantern",
    "jellyfish","kangaroo","knife","ladder","lamp","laptop","leaf","lion","lizard",
    "lobster","microphone","monkey","mosquito","motorcycle","mouse","mushroom","owl",
    "parrot","penguin","piano","pickup_truck","pig","pigeon","pineapple","pistol",
    "pizza","pretzel","rabbit","raccoon","ray","rhinoceros","rifle","rocket","rooster",
    "sailboat","saw","saxophone","scorpion","sea_turtle","seagull","seahorse","shark",
    "sheep","shoe","skyscraper","snail","snake","spider","squirrel","starfish",
    "strawberry","swan","sword","table","tank","teapot","tiger","toilet","train",
    "trumpet","turtle","umbrella","violin","volcano","wading_bird","wheelchair","windmill",
    "wine_bottle","zebra"
]

def main():
    print("\033[2J\033[H", end="")  # clear screen
    print("=" * 60)
    print("  Sketchy Dataset Sync Monitor  (Ctrl+C to stop)")
    print("=" * 60)

    while True:
        p_cats, p_files = count(PHOTO_DIR,  "jpg")
        s_cats, s_files = count(SKETCH_DIR, "png")

        p_pct = p_cats / TARGET_CATS * 100
        s_pct = s_cats / TARGET_CATS * 100

        # Move cursor to top (no flicker)
        print("\033[3;0H", end="")

        print(f"\n  {'Category':<10} {'Progress':<40} {'Files':>8}")
        print(f"  {'-'*60}")
        print(f"  {'sketch':<10} {bar(s_cats, TARGET_CATS):<40} {s_files:>8,}")
        print(f"  {'photo':<10} {bar(p_cats, TARGET_CATS):<40} {p_files:>8,}")
        print(f"\n  sketch: {s_pct:.1f}% complete   photo: {p_pct:.1f}% complete")

        if p_cats < TARGET_CATS:
            miss = missing(PHOTO_DIR, ALL_CATEGORIES)
            print(f"\n  ⏳ Missing photo categories ({len(miss)}):")
            for i, cat in enumerate(miss[:10]):
                print(f"     - {cat}")
            if len(miss) > 10:
                print(f"     ... and {len(miss)-10} more")
        else:
            print(f"\n  ✅ All {TARGET_CATS} photo categories synced!")

        if s_cats < TARGET_CATS:
            miss = missing(SKETCH_DIR, ALL_CATEGORIES)
            print(f"\n  ⏳ Missing sketch categories ({len(miss)}):")
            for i, cat in enumerate(miss[:10]):
                print(f"     - {cat}")
            if len(miss) > 10:
                print(f"     ... and {len(miss)-10} more")
        else:
            print(f"\n  ✅ All {TARGET_CATS} sketch categories synced!")

        if p_cats >= TARGET_CATS and s_cats >= TARGET_CATS:
            print("\n" + "=" * 60)
            print("  🎉 SYNC COMPLETE! Run: python verify_dataset.py")
            print("=" * 60)
            break

        print(f"\n  Refreshing every 30s ... (last check: {time.strftime('%H:%M:%S')})")
        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
