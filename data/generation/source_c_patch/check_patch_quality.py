#!/usr/bin/env python3
"""Massive parallel quality check for 10M+ source C patch images.

Checks for: file corruption, incomplete reads, format issues, dimension errors,
encoding issues, blank/uniform images, and extreme brightness/darkness.

Usage:
    python check_patch_quality.py --dataset source C-colorectal-b2  # Single dataset
    python check_patch_quality.py --dataset source C-colorectal-b2 --workers 2  # Fewer workers
    python check_patch_quality.py --dataset source C-colorectal-b2 --no-pixel-checks  # Skip Tier 4
    python check_patch_quality.py --report-only  # Summarize all results
"""

import os
import sys
import csv
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageFile

# Force PIL to raise on truncated images instead of silently tolerating them
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Import config from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from config import PATCH_SIZE, OUTPUT_BASE as PATCH_BASE, DATASETS

EXPECTED_DIMS = (PATCH_SIZE, PATCH_SIZE)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MIN_FILE_SIZE = 1024  # bytes

# Pixel check thresholds
BLANK_STD_THRESHOLD = 2.0
EXTREME_BRIGHT_THRESHOLD = 250
EXTREME_DARK_THRESHOLD = 5

# Error categories
CAT_EMPTY_FILE = "empty_file"
CAT_SUSPICIOUS_SIZE = "suspiciously_small"
CAT_FORMAT_MISMATCH = "format_mismatch"
CAT_DIMENSION_ERROR = "dimension_error"
CAT_MODE_ERROR = "mode_error"
CAT_HEADER_CORRUPT = "header_corrupt"
CAT_DECODE_ERROR = "decode_error"
CAT_TRUNCATED = "truncated"
CAT_ENCODING_ERROR = "encoding_error"
CAT_BLANK_UNIFORM = "blank_uniform"
CAT_EXTREME_BRIGHTNESS = "extreme_brightness"
CAT_EXTREME_DARKNESS = "extreme_darkness"
CAT_LISTDIR_ERROR = "listdir_error"
CAT_NON_JPEG = "non_jpeg_file"

CSV_FIELDS = ['filepath', 'dataset', 'case', 'error_category', 'error_detail', 'file_size']


def check_single_file(filepath, do_pixel_checks=True):
    """Run all quality tiers on a single image file. Returns list of error dicts."""
    errors = []
    fname = os.path.basename(filepath)
    ext = os.path.splitext(fname)[1].lower()

    # Tier 1: Filesystem metadata (no file read)

    # Flag non-JPEG files (only JPEG allowed after conversion)
    if ext not in ('.jpg', '.jpeg'):
        errors.append({
            'filepath': filepath, 'error_category': CAT_NON_JPEG,
            'error_detail': f"expected JPEG, got {ext} file", 'file_size': -1
        })
        # Still continue checking - the file may also be corrupt

    try:
        fsize = os.path.getsize(filepath)
    except OSError as e:
        errors.append({
            'filepath': filepath, 'error_category': CAT_HEADER_CORRUPT,
            'error_detail': f"stat failed: {e}", 'file_size': -1
        })
        return errors

    # Update non-JPEG error with actual file size
    if errors and errors[0]['error_category'] == CAT_NON_JPEG:
        errors[0]['file_size'] = fsize

    if fsize == 0:
        errors.append({
            'filepath': filepath, 'error_category': CAT_EMPTY_FILE,
            'error_detail': "file is 0 bytes", 'file_size': 0
        })
        return errors  # No point checking further

    if fsize < MIN_FILE_SIZE:
        errors.append({
            'filepath': filepath, 'error_category': CAT_SUSPICIOUS_SIZE,
            'error_detail': f"file is only {fsize} bytes (min {MIN_FILE_SIZE})", 'file_size': fsize
        })
        return errors  # Too small to be a valid 1024x1024 image

    # Tier 2: PIL header check (no full decode)
    expected_format = 'JPEG' if ext in ('.jpg', '.jpeg') else 'PNG'

    try:
        img = Image.open(filepath)
    except Exception as e:
        errors.append({
            'filepath': filepath, 'error_category': CAT_HEADER_CORRUPT,
            'error_detail': f"Image.open() failed: {type(e).__name__}: {e}", 'file_size': fsize
        })
        return errors

    if img.format != expected_format:
        errors.append({
            'filepath': filepath, 'error_category': CAT_FORMAT_MISMATCH,
            'error_detail': f"expected {expected_format}, got {img.format}", 'file_size': fsize
        })

    if img.size != EXPECTED_DIMS:
        errors.append({
            'filepath': filepath, 'error_category': CAT_DIMENSION_ERROR,
            'error_detail': f"expected {EXPECTED_DIMS}, got {img.size}", 'file_size': fsize
        })

    if img.mode != 'RGB':
        errors.append({
            'filepath': filepath, 'error_category': CAT_MODE_ERROR,
            'error_detail': f"expected RGB, got {img.mode}", 'file_size': fsize
        })

    # If header is badly broken, skip decode
    if any(e['error_category'] == CAT_HEADER_CORRUPT for e in errors):
        try:
            img.close()
        except Exception:
            pass
        return errors

    # Tier 3: Full pixel decode
    try:
        img.load()
    except OSError as e:
        err_str = str(e).lower()
        if 'truncated' in err_str:
            cat = CAT_TRUNCATED
        else:
            cat = CAT_ENCODING_ERROR
        errors.append({
            'filepath': filepath, 'error_category': cat,
            'error_detail': f"load() failed: {type(e).__name__}: {e}", 'file_size': fsize
        })
        try:
            img.close()
        except Exception:
            pass
        return errors
    except Exception as e:
        errors.append({
            'filepath': filepath, 'error_category': CAT_DECODE_ERROR,
            'error_detail': f"load() failed: {type(e).__name__}: {e}", 'file_size': fsize
        })
        try:
            img.close()
        except Exception:
            pass
        return errors

    # Tier 4: Pixel statistics (only if decode succeeded)
    if do_pixel_checks:
        import numpy as np
        try:
            arr = np.array(img, dtype=np.float32)
            pixel_mean = float(arr.mean())
            pixel_std = float(arr.std())

            if pixel_std < BLANK_STD_THRESHOLD:
                errors.append({
                    'filepath': filepath, 'error_category': CAT_BLANK_UNIFORM,
                    'error_detail': f"std={pixel_std:.2f} (threshold {BLANK_STD_THRESHOLD})", 'file_size': fsize
                })

            if pixel_mean > EXTREME_BRIGHT_THRESHOLD:
                errors.append({
                    'filepath': filepath, 'error_category': CAT_EXTREME_BRIGHTNESS,
                    'error_detail': f"mean={pixel_mean:.1f} (threshold {EXTREME_BRIGHT_THRESHOLD})", 'file_size': fsize
                })
            elif pixel_mean < EXTREME_DARK_THRESHOLD:
                errors.append({
                    'filepath': filepath, 'error_category': CAT_EXTREME_DARKNESS,
                    'error_detail': f"mean={pixel_mean:.1f} (threshold {EXTREME_DARK_THRESHOLD})", 'file_size': fsize
                })
        except Exception as e:
            errors.append({
                'filepath': filepath, 'error_category': CAT_DECODE_ERROR,
                'error_detail': f"pixel stats failed: {type(e).__name__}: {e}", 'file_size': fsize
            })

    try:
        img.close()
    except Exception:
        pass

    return errors


def check_slide_dir(slide_dir, dataset_name, do_pixel_checks=True):
    """Check all image files in a single slide directory."""
    case_name = os.path.basename(slide_dir)

    try:
        files = os.listdir(slide_dir)
    except OSError as e:
        return {
            'slide_dir': slide_dir, 'case': case_name,
            'n_checked': 0, 'n_passed': 0, 'n_bad': 0,
            'bad_records': [{
                'filepath': slide_dir, 'dataset': dataset_name,
                'case': case_name, 'error_category': CAT_LISTDIR_ERROR,
                'error_detail': f"os.listdir failed: {e}", 'file_size': -1
            }],
            'elapsed': 0
        }

    # Filter to image files only
    image_files = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)

    if not image_files:
        return {
            'slide_dir': slide_dir, 'case': case_name,
            'n_checked': 0, 'n_passed': 0, 'n_bad': 0,
            'bad_records': [], 'elapsed': 0
        }

    t0 = time.time()
    bad_records = []
    n_bad = 0

    for fname in image_files:
        fpath = os.path.join(slide_dir, fname)
        errs = check_single_file(fpath, do_pixel_checks=do_pixel_checks)
        if errs:
            n_bad += 1
            for e in errs:
                e['dataset'] = dataset_name
                e['case'] = case_name
                bad_records.append(e)

    n_checked = len(image_files)
    elapsed = time.time() - t0

    return {
        'slide_dir': slide_dir, 'case': case_name,
        'n_checked': n_checked, 'n_passed': n_checked - n_bad,
        'n_bad': n_bad, 'bad_records': bad_records, 'elapsed': elapsed
    }


def _worker(args):
    """Multiprocessing worker wrapper."""
    slide_dir, dataset_name, do_pixel_checks = args
    return check_slide_dir(slide_dir, dataset_name, do_pixel_checks)


def load_progress(progress_path):
    """Load set of completed directory names from progress file."""
    if not os.path.exists(progress_path):
        return set()
    try:
        with open(progress_path, 'r') as f:
            data = json.load(f)
        return set(data.get('completed_dirs', []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_progress(progress_path, completed_dirs, dataset_name, total_files_checked, total_bad):
    """Atomically save progress file."""
    data = {
        'dataset': dataset_name,
        'last_updated': datetime.now().isoformat(),
        'completed_dirs': sorted(completed_dirs),
        'total_dirs': len(completed_dirs),
        'total_files_checked': total_files_checked,
        'total_bad_files': total_bad
    }
    tmp_path = progress_path + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        os.rename(tmp_path, progress_path)
    except OSError:
        # If rename fails (e.g., cross-device), try direct write
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        with open(progress_path, 'w') as f:
            json.dump(data, f)


def report_mode(output_dir):
    """Read all bad_files CSVs and print a summary report."""
    import glob

    csv_files = sorted(glob.glob(os.path.join(output_dir, 'bad_files_*.csv')))

    if not csv_files:
        print("No bad_files CSVs found in", output_dir)
        return

    print("=" * 80)
    print("source C Patch Quality Check - Summary Report")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Per-dataset breakdown
    all_errors = []
    dataset_stats = {}

    for csv_path in csv_files:
        dataset_name = os.path.basename(csv_path).replace('bad_files_', '').replace('.csv', '')
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_errors.append(row)
                count += 1
        dataset_stats[dataset_name] = count

    # Also read progress files for total counts
    progress_files = sorted(glob.glob(os.path.join(output_dir, 'qc_progress_*.json')))
    total_checked = 0
    progress_data = {}
    for pf in progress_files:
        ds = os.path.basename(pf).replace('qc_progress_', '').replace('.json', '')
        try:
            with open(pf, 'r') as f:
                pdata = json.load(f)
            progress_data[ds] = pdata
            total_checked += pdata.get('total_files_checked', 0)
        except (json.JSONDecodeError, OSError):
            pass

    print("Per-Dataset Breakdown:")
    header = f"  {'Dataset':<30} {'Files Checked':>14} {'Bad Files':>10} {'Fail%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds in sorted(dataset_stats.keys()):
        checked = progress_data.get(ds, {}).get('total_files_checked', '?')
        bad = dataset_stats[ds]
        if isinstance(checked, int) and checked > 0:
            fail_pct = f"{100.0 * bad / checked:.4f}%"
        else:
            fail_pct = "?"
        print(f"  {ds:<30} {str(checked):>14} {bad:>10} {fail_pct:>8}")

    print()

    # Error category summary
    cat_counts = {}
    for row in all_errors:
        cat = row.get('error_category', 'unknown')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("Error Category Summary:")
    print(f"  {'Category':<30} {'Count':>10}")
    print("  " + "-" * 42)
    for cat in sorted(cat_counts.keys(), key=lambda c: -cat_counts[c]):
        print(f"  {cat:<30} {cat_counts[cat]:>10}")

    total_bad = len(set(r.get('filepath', '') for r in all_errors))
    total_error_rows = len(all_errors)

    print()
    print(f"Total files checked: {total_checked:,}")
    print(f"Total bad files: {total_bad:,}")
    print(f"Total error entries: {total_error_rows:,}")
    if total_checked > 0:
        print(f"Overall fail rate: {100.0 * total_bad / total_checked:.4f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='source C Patch Quality Check - Massively parallel image validation')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (e.g., source C-colorectal-b2). Required unless --report-only.')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'logs'),
                        help='Output directory for CSV and progress files')
    parser.add_argument('--no-pixel-checks', action='store_true',
                        help='Skip Tier 4 pixel statistics (faster, structural checks only)')
    parser.add_argument('--report-only', action='store_true',
                        help='Print summary report from existing CSVs and exit')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.report_only:
        report_mode(args.output_dir)
        return

    if args.dataset is None:
        parser.error("--dataset is required unless using --report-only")

    dataset_name = args.dataset
    dataset_dir = os.path.join(PATCH_BASE, dataset_name)

    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    do_pixel_checks = not args.no_pixel_checks
    progress_path = os.path.join(args.output_dir, f"qc_progress_{dataset_name}.json")
    csv_path = os.path.join(args.output_dir, f"bad_files_{dataset_name}.csv")

    # Get all slide directories
    all_slide_dirs = sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    total_dirs = len(all_slide_dirs)

    # Load progress and filter out completed dirs
    completed = load_progress(progress_path)
    remaining_dirs = [d for d in all_slide_dirs if os.path.basename(d) not in completed]
    already_done = total_dirs - len(remaining_dirs)

    print("=" * 70)
    print(f"source C Patch Quality Check: {dataset_name}")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Total slide dirs: {total_dirs}")
    print(f"Already checked: {already_done}")
    print(f"Remaining: {len(remaining_dirs)}")
    print(f"Workers: {args.workers}")
    print(f"Pixel checks: {'ON' if do_pixel_checks else 'OFF'}")
    print(f"Output CSV: {csv_path}")
    print(f"Progress file: {progress_path}")
    print("=" * 70)

    if not remaining_dirs:
        print("All directories already checked. Nothing to do.")
        print("Use --report-only for summary, or delete progress file to re-check.")
        return

    # Open CSV for append
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()

    total_checked = 0
    total_passed = 0
    total_bad = 0
    dirs_done = 0
    start_time = time.time()
    progress_batch = 0

    worker_args = [(sd, dataset_name, do_pixel_checks) for sd in remaining_dirs]

    with mp.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_worker, worker_args):
            dirs_done += 1
            total_checked += result['n_checked']
            total_passed += result['n_passed']
            total_bad += result['n_bad']

            # Write bad records immediately
            if result['bad_records']:
                for rec in result['bad_records']:
                    csv_writer.writerow(rec)
                csv_file.flush()

            # Track completed dir
            completed.add(result['case'])
            progress_batch += 1

            # Save progress every 10 dirs
            if progress_batch >= 10:
                save_progress(progress_path, completed, dataset_name, total_checked, total_bad)
                progress_batch = 0

            # Print progress
            n_bad = result['n_bad']
            if n_bad > 0 or dirs_done % 50 == 0:
                pct = 100.0 * dirs_done / len(remaining_dirs)
                elapsed = time.time() - start_time
                rate = total_checked / elapsed if elapsed > 0 else 0
                eta_min = (len(remaining_dirs) - dirs_done) / (dirs_done / elapsed / 60) if dirs_done > 0 and elapsed > 0 else 0
                print(f"  [{dirs_done}/{len(remaining_dirs)} = {pct:.1f}%] "
                      f"{result['case']}: checked {result['n_checked']}, bad {n_bad} | "
                      f"Total: {total_checked:,} checked, {total_bad} bad | "
                      f"{rate:.0f} img/s | ETA ~{eta_min:.0f}min")

    csv_file.close()

    # Final progress save
    save_progress(progress_path, completed, dataset_name, total_checked, total_bad)

    elapsed = time.time() - start_time
    rate = total_checked / elapsed if elapsed > 0 else 0

    print()
    print("=" * 70)
    print(f"DONE: {dataset_name}")
    print(f"  Slide dirs checked: {dirs_done:,}")
    print(f"  Images checked: {total_checked:,}")
    print(f"  Passed: {total_passed:,}")
    print(f"  Bad files: {total_bad:,}")
    if total_checked > 0:
        print(f"  Fail rate: {100.0 * total_bad / total_checked:.4f}%")
    print(f"  Time: {elapsed/3600:.2f} hours ({elapsed:.1f}s)")
    print(f"  Rate: {rate:.0f} images/sec")
    print(f"  Workers: {args.workers}")
    print(f"  Bad files CSV: {csv_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
