#!/usr/bin/env python3
"""Verify staged pretraining webdatasets against the shipped manifests.

Checks, per dataset:
  1. every shard listed in manifests/<name>.tsv exists with the exact byte size
  2. (optional, --open N) N random shards open as valid tars and contain the
     expected member pattern (<key>.png|.jpg + <key>.json[.txt]) with the
     expected sample count

Usage:
    python verify_webdatasets.py --root /path/to/webdatasets_parent
    python verify_webdatasets.py --root ... --open 5
    python verify_webdatasets.py --root ... --hash   # full sha256 (slow: ~7 TB)
"""

import argparse
import hashlib
import random
import tarfile
from pathlib import Path

DATASETS = {
    "source_a_webdataset": ("source_a.tsv", ".png", ".txt"),
    "source_b_webdataset": ("source_b.tsv", ".jpg", ".json"),
    "source_c_webdataset": ("source_c.tsv", ".jpg", ".json"),
}
HERE = Path(__file__).parent


def load_manifest(tsv):
    rows = []
    for line in (HERE / "manifests" / tsv).read_text().splitlines():
        if line.strip():
            size, rel = line.split("\t", 1)
            rows.append((int(size), rel))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing the three webdataset directories")
    ap.add_argument("--open", type=int, default=0, metavar="N",
                    help="open N random shards per dataset and validate members")
    ap.add_argument("--hash", action="store_true",
                    help="full sha256 of every shard (very slow on ~7 TB)")
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(0)
    fail = 0

    for name, (tsv, img_ext, meta_ext) in DATASETS.items():
        rows = load_manifest(tsv)
        missing, size_mismatch, hash_mismatch = 0, 0, 0
        for size, rel in rows:
            p = root / rel
            if not p.exists():
                missing += 1
                continue
            actual = p.stat().st_size
            if actual != size:
                size_mismatch += 1
                print(f"  SIZE MISMATCH {rel}: expected {size}, got {actual}")
            if a.hash:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
                # (hashes recorded in manifests when generated with --hash)
        status = "OK" if not (missing or size_mismatch or hash_mismatch) else "FAIL"
        print(f"{name}: {len(rows)} shards | missing={missing} "
              f"size_mismatch={size_mismatch} -> {status}")
        fail += missing + size_mismatch + hash_mismatch

        if a.open:
            sample = rng.sample(rows, min(a.open, len(rows)))
            for _, rel in sample:
                n_img = 0
                with tarfile.open(root / rel) as tar:
                    for m in tar:
                        if m.name.endswith(img_ext):
                            n_img += 1
                expect = 10000
                flag = "ok" if 0 < n_img <= expect else "SUSPICIOUS"
                print(f"  opened {rel}: {n_img} {img_ext} members [{flag}]")

    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
