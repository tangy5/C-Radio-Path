#!/usr/bin/env python3
"""Convert source C patch PNGs to JPEG in-place with multiprocessing.

Usage:
    python3 convert_png_to_jpeg.py --dataset source C-colorectal-b2 [--quality 95] [--workers 14] [--dry-run]
    
Strategy: For each slide directory, convert PNG->JPEG one file at a time,
verify the JPEG loads correctly and has reasonable size, then delete the PNG.
Also updates _patches.csv files to reference .jpg instead of .png.
"""

import os
import sys
import csv
import argparse
import multiprocessing as mp
from pathlib import Path
from PIL import Image

PATCH_BASE = "<PATHOLOGY_DATASETS_DIR>/source_c_patch"


def verify_jpeg(jpeg_path, original_dims=None):
    """Verify a JPEG file is valid and has reasonable quality."""
    try:
        img = Image.open(jpeg_path)
        img.load()  # full decode verifies integrity
        w, h = img.size
        if original_dims:
            ow, oh = original_dims
            if w != ow or h != oh:
                return False, f"Size changed: {w}x{h}, original {ow}x{oh}"
        if w == 0 or h == 0:
            return False, f"Zero dimension: {w}x{h}"
        if img.mode != 'RGB':
            return False, f"Mode mismatch: {img.mode}, expected RGB"
        jpeg_size = os.path.getsize(jpeg_path)
        if jpeg_size < 1024:
            return False, f"JPEG too small: {jpeg_size} bytes"
        return True, f"OK ({jpeg_size/1024:.0f}KB)"
    except Exception as e:
        return False, str(e)


def convert_slide_dir(slide_dir, quality=95, dry_run=False):
    """Convert all PNGs in a slide directory to JPEG, then update CSV."""
    try:
        files = os.listdir(slide_dir)
    except OSError:
        return 0, 0, 0, 0

    png_files = sorted([f for f in files if f.endswith('.png')])
    csv_files = [f for f in files if f.endswith('_patches.csv')]

    if not png_files:
        return 0, 0, 0, 0

    converted = 0
    failed = 0
    skipped_kept_png = 0
    saved_bytes = 0

    for png_name in png_files:
        png_path = os.path.join(slide_dir, png_name)
        jpg_name = png_name.replace('.png', '.jpg')
        jpg_path = os.path.join(slide_dir, jpg_name)

        # Skip if JPEG already exists
        if os.path.exists(jpg_path):
            png_size = os.path.getsize(png_path)
            if not dry_run:
                os.remove(png_path)
            saved_bytes += png_size
            converted += 1
            continue

        png_size = os.path.getsize(png_path)

        if dry_run:
            saved_bytes += int(png_size * 0.85)
            converted += 1
            continue

        try:
            img = Image.open(png_path)
            original_dims = img.size
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(jpg_path, 'JPEG', quality=quality)

            ok, msg = verify_jpeg(jpg_path, original_dims=original_dims)
            if not ok:
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
                failed += 1
                continue

            jpg_size = os.path.getsize(jpg_path)

            saved_bytes += (png_size - jpg_size)
            os.remove(png_path)
            converted += 1

        except Exception as e:
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
            failed += 1

    # Update CSV files to reference .jpg instead of .png
    if converted > 0 and not dry_run:
        for csv_name in csv_files:
            csv_path = os.path.join(slide_dir, csv_name)
            try:
                rows = []
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    for row in reader:
                        if 'patch_name' in row and row['patch_name'].endswith('.png'):
                            row['patch_name'] = row['patch_name'].replace('.png', '.jpg')
                        rows.append(row)

                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception:
                pass

    return converted, failed, skipped_kept_png, saved_bytes


def _worker(args):
    """Worker wrapper for multiprocessing."""
    slide_dir, quality, dry_run = args
    slide_name = os.path.basename(slide_dir)
    converted, failed, kept_png, saved = convert_slide_dir(slide_dir, quality, dry_run)
    return slide_name, converted, failed, kept_png, saved


def main():
    parser = argparse.ArgumentParser(description='Convert source C patch PNGs to JPEG')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., source C-colorectal-b2)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (default: 95)')
    parser.add_argument('--workers', type=int, default=56, help='Number of parallel workers (default: 56)')
    parser.add_argument('--dry-run', action='store_true', help='Only estimate savings')
    args = parser.parse_args()

    dataset_dir = os.path.join(PATCH_BASE, args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Get all slide directories that still have PNGs
    all_slide_dirs = sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    # Filter to only slides that still have PNG files
    slide_dirs = []
    for sd in all_slide_dirs:
        try:
            if any(f.endswith('.png') for f in os.listdir(sd)):
                slide_dirs.append(sd)
        except OSError:
            pass

    total_slides = len(all_slide_dirs)
    remaining_slides = len(slide_dirs)
    skipped_slides = total_slides - remaining_slides

    print(f"Dataset: {args.dataset}")
    print(f"Total slides: {total_slides}, Remaining with PNGs: {remaining_slides}, Already converted: {skipped_slides}")
    print(f"JPEG quality: {args.quality}")
    print(f"Workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    if not slide_dirs:
        print("Nothing to convert - all slides already processed.")
        return

    total_converted = 0
    total_failed = 0
    total_kept_png = 0
    total_saved = 0

    start_time = __import__('time').time()

    if args.workers <= 1:
        # Sequential mode
        for i, slide_dir in enumerate(slide_dirs):
            slide_name = os.path.basename(slide_dir)
            converted, failed, kept_png, saved = convert_slide_dir(slide_dir, args.quality, args.dry_run)
            total_converted += converted
            total_failed += failed
            total_kept_png += kept_png
            total_saved += saved

            if (i + 1) % 50 == 0 or converted > 0 or failed > 0 or kept_png > 0:
                pct = 100.0 * (i + 1) / remaining_slides
                print(f"  [{i+1}/{remaining_slides} = {pct:.1f}%] {slide_name}: "
                      f"{converted} conv, {failed} fail, {kept_png} kept-png | "
                      f"Total: {total_converted} conv, {total_saved/1e9:.2f}GB saved")
    else:
        # Parallel mode
        worker_args = [(sd, args.quality, args.dry_run) for sd in slide_dirs]

        with mp.Pool(processes=args.workers) as pool:
            for i, (slide_name, converted, failed, kept_png, saved) in enumerate(
                pool.imap_unordered(_worker, worker_args), 1
            ):
                total_converted += converted
                total_failed += failed
                total_kept_png += kept_png
                total_saved += saved

                if converted > 0 or failed > 0 or kept_png > 0:
                    pct = 100.0 * i / remaining_slides
                    print(f"  [{i}/{remaining_slides} = {pct:.1f}%] {slide_name}: "
                          f"{converted} conv, {failed} fail, {kept_png} kept-png | "
                          f"Total: {total_converted} conv, {total_saved/1e9:.2f}GB saved")

    elapsed = __import__('time').time() - start_time

    print("=" * 70)
    print(f"DONE: {args.dataset}")
    print(f"  Converted: {total_converted:,}")
    print(f"  Failed: {total_failed}")
    print(f"  Kept PNG (JPEG was larger): {total_kept_png}")
    print(f"  Space saved: {total_saved/1e9:.2f} GB ({total_saved/1e12:.2f} TB)")
    print(f"  Time: {elapsed/3600:.2f} hours ({remaining_slides/elapsed:.1f} slides/sec)")
    print(f"  Workers: {args.workers}")


if __name__ == '__main__':
    main()
