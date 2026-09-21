#!/usr/bin/env python3
"""
Construct a source C webdataset from quality-checked patches.

Matches source B webdataset spec:
- Image-only samples: __key__, jpg (image bytes), json (sha256)
- 10,000 samples per shard
- Max 1000 tarfiles per subdirectory
- Shuffled with seed=42
- Automatically excludes bad patches identified by check_patch_quality.py

Usage:
    python construct_source_c_webdataset.py                           # Convert all datasets
    python construct_source_c_webdataset.py --dataset source C-breast   # Single dataset
    python construct_source_c_webdataset.py --workers 64 --shard-size 10000
"""

import os
import sys
import csv
import json
import glob
import hashlib
import random
import argparse
import multiprocessing as mp
from pathlib import Path

import webdataset as wds

# Config matching source_b1M specs
SHARD_SIZE = 10000
MAX_PER_SUBDIR = 1000
SEED = 42
IMAGE_EXTENSIONS = {'.jpg', '.jpeg'}  # JPEG only for source C (PNGs are flagged as bad)

SOURCE_C_PATCH = "<PATHOLOGY_DATASETS_DIR>/source_c_patch"
LOGS_DIR = os.path.join(SOURCE_C_PATCH, "logs")
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(SOURCE_C_PATCH), "source_c_webdataset"
)

DATASETS = [
    "source C-breast",
    "source C-colorectal-b1",
    "source C-colorectal-b2",
    "source C-gastrointestinal",
    "source C-hematologic",
    "source C-skin-b1",
    "source C-skin-b2",
    "source C-thorax",
]


def compute_sha256(data: bytes) -> str:
    m = hashlib.sha256()
    m.update(data)
    return m.hexdigest()


def load_bad_files(dataset_name: str = None):
    """Load set of bad filepaths from QC CSVs."""
    if dataset_name:
        csv_pattern = os.path.join(LOGS_DIR, f"bad_files_{dataset_name}.csv")
    else:
        csv_pattern = os.path.join(LOGS_DIR, "bad_files_*.csv")

    bad = set()
    for csv_path in sorted(glob.glob(csv_pattern)):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fp = row.get('filepath', '').strip()
                    if fp:
                        bad.add(fp)
        except Exception:
            pass
    return bad


def find_images(dataset_dir, bad_files):
    """Find all JPEG images in a dataset directory, excluding bad files."""
    images = []
    skipped_bad = 0
    skipped_non_jpeg = 0

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            ext = Path(fname).suffix.lower()
            fpath = os.path.join(root, fname)

            if fpath in bad_files:
                skipped_bad += 1
                continue

            if ext in IMAGE_EXTENSIONS:
                images.append(fpath)
            elif ext in {'.png'}:
                skipped_non_jpeg += 1

    return images, skipped_bad, skipped_non_jpeg


def process_image(args):
    """Process a single image: read, hash, return sample dict or None."""
    image_path, idx = args
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
    except Exception:
        return None

    sha256_hash = compute_sha256(image_data)
    ext = Path(image_path).suffix.lower().lstrip('.')
    if ext == 'jpeg':
        ext = 'jpg'

    key = f"{idx:09d}"

    return {
        "__key__": key,
        ext: image_data,
        "json": json.dumps({"sha256": sha256_hash}).encode('utf-8'),
    }


def get_subdir_for_shard(shard_idx, max_per_subdir=MAX_PER_SUBDIR):
    subdir_idx = (shard_idx // max_per_subdir) + 1
    return str(subdir_idx * max_per_subdir)


def validate_shard(shard_path, expected_samples):
    """Check if an existing shard has the expected number of valid samples.
    Uses fast file-size heuristic first, then falls back to tarfile validation.
    Returns (is_valid, actual_samples)."""
    if not os.path.exists(shard_path):
        return False, 0

    # Fast path: check file size (expected ~5.5-5.7 GB for 10k samples)
    size = os.path.getsize(shard_path)
    min_expected_size = 5_000_000_000  # ~5 GB lower bound
    max_expected_size = 6_000_000_000  # ~6 GB upper bound

    if expected_samples == 10000 and min_expected_size <= size <= max_expected_size:
        return True, expected_samples

    # Slow path: full tarfile validation for unexpected sizes
    try:
        import tarfile
        tf = tarfile.open(shard_path, 'r')
        members = tf.getmembers()
        jpg_count = sum(1 for m in members if m.name.endswith('.jpg'))
        json_count = sum(1 for m in members if m.name.endswith('.json'))
        if jpg_count == json_count == expected_samples:
            return True, jpg_count
        return False, jpg_count
    except Exception:
        return False, 0


def write_shards(image_paths, output_dir, shard_size=SHARD_SIZE,
                 max_per_subdir=MAX_PER_SUBDIR, num_workers=16,
                 start_idx=0):
    """Write image files to webdataset shards using batched multiprocessing.
    
    Processes one shard at a time to avoid overwhelming the multiprocessing
    queue with 10M+ items. Automatically skips valid existing shards on resume.
    """
    import time
    n = len(image_paths)
    num_shards = (n + shard_size - 1) // shard_size
    num_subdirs = (num_shards + max_per_subdir - 1) // max_per_subdir

    print(f"  Images: {n:,}")
    print(f"  Shards: ~{num_shards} ({shard_size} per shard)")
    print(f"  Subdirs: ~{num_subdirs}")
    print(f"  Workers: {num_workers}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    failed = 0
    skipped_shards = 0
    t0 = time.time()

    with mp.Pool(processes=num_workers) as pool:
        for shard_idx in range(num_shards):
            batch_start = shard_idx * shard_size
            batch_end = min(batch_start + shard_size, n)
            batch_size = batch_end - batch_start
            
            # Determine shard path
            subdir = get_subdir_for_shard(shard_idx, max_per_subdir)
            shard_dir = os.path.join(output_dir, subdir)
            shard_path = os.path.join(shard_dir, f"shard_{shard_idx:06d}.tar")
            
            # Resume: skip if shard already valid
            is_valid, existing = validate_shard(shard_path, batch_size)
            if is_valid:
                successful += existing
                skipped_shards += 1
                continue
            elif existing > 0:
                # Corrupt/incomplete shard from previous run — delete and redo
                print(f"    [shard {shard_idx+1}/{num_shards}] Corrupt ({existing}/{batch_size} samples), redoing...", flush=True)
                os.remove(shard_path)
            
            batch_paths = image_paths[batch_start:batch_end]
            
            # Build work items for this batch only
            work_items = [(batch_paths[i], start_idx + batch_start + i)
                          for i in range(len(batch_paths))]

            # Process batch in parallel
            results = pool.map(process_image, work_items, chunksize=50)

            # Ensure shard directory exists
            os.makedirs(shard_dir, exist_ok=True)

            # Write results to shard
            batch_ok = 0
            with wds.TarWriter(shard_path) as writer:
                for result in results:
                    if result is None:
                        failed += 1
                        continue
                    writer.write(result)
                    batch_ok += 1
                    successful += 1

            # Progress reporting
            elapsed = time.time() - t0
            if elapsed > 0:
                rate = successful / elapsed
                remaining = n - successful
                eta = remaining / rate / 60 if rate > 0 else 0
            else:
                rate = 0
                eta = 0
            pct = 100 * successful / n
            skip_info = f" [+{skipped_shards} skipped]" if skipped_shards else ""
            print(f"    [shard {shard_idx+1}/{num_shards}]{skip_info} "
                  f"[{successful:,}/{n:,} = {pct:.1f}%] "
                  f"{rate:.0f} img/s | ETA ~{eta:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: {successful:,} written, {failed} failed, {skipped_shards} shards skipped")
    print(f"  Time: {elapsed/60:.1f} min ({n/elapsed:.0f} img/s overall)", flush=True)
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Build source C webdataset from quality-checked patches'
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name (default: all 8)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output directory (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--shard-size', type=int, default=SHARD_SIZE,
                        help=f'Samples per shard (default: {SHARD_SIZE})')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers for image processing')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed for shuffling (default: {SEED})')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Skip shuffling (process in filesystem order)')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS

    print("=" * 70)
    print("source C Patch Webdataset Construction")
    print("=" * 70)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Output: {args.output}")
    print(f"Shard size: {args.shard_size}")
    print(f"Workers: {args.workers}")
    print(f"Seed: {args.seed}" if not args.no_shuffle else "Shuffle: OFF")
    print("=" * 70)

    # Load all bad files upfront
    all_bad = load_bad_files()
    print(f"\nExcluding {len(all_bad):,} known bad patches from QC")

    total_images = 0
    total_skipped_bad = 0
    total_skipped_png = 0
    all_image_paths = []

    for ds in datasets:
        ds_dir = os.path.join(SOURCE_C_PATCH, ds)
        if not os.path.isdir(ds_dir):
            print(f"\n  WARNING: Dataset dir not found: {ds_dir}")
            continue

        print(f"\n  Scanning {ds}...")
        imgs, skip_bad, skip_png = find_images(ds_dir, all_bad)
        print(f"    Found {len(imgs):,} clean JPEG images "
              f"({skip_bad:,} bad excluded, {skip_png:,} PNG skipped)")

        all_image_paths.extend(imgs)
        total_images += len(imgs)
        total_skipped_bad += skip_bad
        total_skipped_png += skip_png

    print(f"\n{'='*70}")
    print(f"Total clean JPEG images: {total_images:,}")
    print(f"Total bad patches excluded: {total_skipped_bad:,}")
    print(f"Total PNG skipped (not converted): {total_skipped_png:,}")
    print("=" * 70)

    if not all_image_paths:
        print("No images to convert!")
        return

    # Shuffle (unless --no-shuffle)
    if not args.no_shuffle:
        print("\nShuffling images...")
        random.seed(args.seed)
        random.shuffle(all_image_paths)
    else:
        # Sort by path for deterministic ordering
        all_image_paths.sort()

    # Write shards
    print(f"\nWriting webdataset shards to {args.output}...")
    write_shards(
        all_image_paths, args.output,
        shard_size=args.shard_size,
        num_workers=args.workers,
    )

    print(f"\n{'='*70}")
    print("Webdataset construction complete!")
    print(f"Output: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
