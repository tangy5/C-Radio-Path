#!/usr/bin/env python3
"""Main pipeline to process all source C datasets and extract foreground patches.

Processes each dataset using multiprocessing:
1. Scans for WSI TIFF files
2. Detects tissue foreground using Otsu thresholding
3. Extracts 1024x1024 patches from foreground regions
4. Saves patches as JPEG (or PNG) with coordinate metadata

Usage:
    python run_pipeline.py              # Process all datasets
    python run_pipeline.py --dataset source C-skin-b1  # Process single dataset
    python run_pipeline.py --dry-run    # Show what would be processed
    python run_pipeline.py --workers 14 # Use 14 parallel workers
"""

import os
import sys
import time
import json
import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (SOURCE_C_BASE, OUTPUT_BASE as CONFIG_OUTPUT_BASE, DATASETS, 
                     PATCH_SIZE as CONFIG_PATCH_SIZE, MIN_FOREGROUND_RATIO as CONFIG_MIN_FG, 
                     OVERLAP, SAVE_COORDINATES, CASE_SELECTION_FILE,
                     PATCH_FORMAT as CONFIG_PATCH_FORMAT, 
                     JPEG_QUALITY as CONFIG_JPEG_QUALITY,
                     MAX_WORKERS as CONFIG_MAX_WORKERS)
from patch_extraction import process_wsi


def load_case_selection(selection_file=CASE_SELECTION_FILE):
    """Load selected case IDs from selection file.
    
    Returns dict mapping dataset_name -> set of selected case IDs.
    Returns None if no selection file is configured or file doesn't exist.
    """
    if selection_file is None or not os.path.exists(selection_file):
        return None
    
    try:
        with open(selection_file, 'r') as f:
            data = json.load(f)
        
        selection = {}
        for dataset_name, info in data.get('datasets', {}).items():
            selection[dataset_name] = set(info.get('case_ids', []))
        
        return selection
    except Exception as e:
        print(f"  WARNING: Could not load case selection file: {e}")
        return None


def find_wsi_files(dataset_dir: str, selected_cases: set = None) -> list:
    """Find all WSI TIFF files in a dataset directory."""
    tiff_files = []
    skipped = 0
    
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for f in files:
            if f.lower().endswith(('.tiff', '.tif')):
                full_path = os.path.join(root, f)
                
                if selected_cases is not None:
                    parent_dir = os.path.basename(os.path.dirname(full_path))
                    if parent_dir.startswith('case_'):
                        case_id = parent_dir.replace('case_', '')
                        case_id_normalized = str(int(case_id)) if case_id.isdigit() else case_id
                        
                        if case_id_normalized not in selected_cases and case_id not in selected_cases:
                            skipped += 1
                            continue
                
                tiff_files.append(full_path)
    
    if skipped > 0:
        print(f"  (Skipped {skipped} WSI files not in selection)")
    
    return sorted(tiff_files)


def _process_wsi_worker(tiff_path, output_base, patch_size, min_fg_ratio, overlap,
                        save_coords, patch_format, jpeg_quality):
    """Worker function for multiprocessing. Wraps process_wsi and returns result dict."""
    wsi_name = os.path.basename(os.path.dirname(tiff_path)) + "_" + Path(tiff_path).stem
    
    try:
        num_patches, out_dir = process_wsi(
            tiff_path,
            output_base,
            patch_size=patch_size,
            min_fg_ratio=min_fg_ratio,
            overlap=overlap,
            save_coords=save_coords,
            patch_format=patch_format,
            jpeg_quality=jpeg_quality
        )
        return {'wsi': wsi_name, 'patches': num_patches, 'status': 'ok'}
    except Exception as e:
        return {'wsi': wsi_name, 'patches': 0, 'status': 'failed', 'error': str(e)}


def process_dataset(dataset_name: str, dry_run: bool = False, 
                    case_selection: dict = None,
                    workers: int = 1,
                    patch_size: int = CONFIG_PATCH_SIZE,
                    min_fg_ratio: float = CONFIG_MIN_FG,
                    patch_format: str = CONFIG_PATCH_FORMAT,
                    jpeg_quality: int = CONFIG_JPEG_QUALITY) -> dict:
    """Process a single dataset with multiprocessing support."""
    dataset_dir = os.path.join(SOURCE_C_BASE, dataset_name)
    dataset_output = os.path.join(CONFIG_OUTPUT_BASE, dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f"  WARNING: Dataset directory not found: {dataset_dir}")
        return {'wsi_count': 0, 'patch_count': 0, 'status': 'missing'}
    
    selected_cases = case_selection.get(dataset_name) if case_selection else None
    
    if selected_cases is not None:
        print(f"  Using case selection: {len(selected_cases)} cases selected")
    
    print(f"  Scanning for WSI files...")
    tiff_files = find_wsi_files(dataset_dir, selected_cases)
    
    if not tiff_files:
        print(f"  No TIFF files found in {dataset_dir}")
        return {'wsi_count': 0, 'patch_count': 0, 'status': 'no_files'}
    
    print(f"  Found {len(tiff_files)} WSI files")
    
    if selected_cases is not None:
        coverage = len(tiff_files) / len(selected_cases) * 100 if len(selected_cases) > 0 else 0
        print(f"  (Coverage: {len(tiff_files)}/{len(selected_cases)} selected cases = {coverage:.1f}%)")
    
    if dry_run:
        print(f"  [DRY RUN] Would process {len(tiff_files)} WSIs with {workers} workers")
        for tiff in tiff_files[:3]:
            print(f"    - {tiff}")
        if len(tiff_files) > 3:
            print(f"    ... and {len(tiff_files) - 3} more")
        return {'wsi_count': len(tiff_files), 'patch_count': 0, 'status': 'dry_run'}
    
    total_patches = 0
    processed = 0
    failed = 0
    start_time = time.time()
    total_wsis = len(tiff_files)
    
    worker_fn = partial(
        _process_wsi_worker,
        output_base=dataset_output,
        patch_size=patch_size,
        min_fg_ratio=min_fg_ratio,
        overlap=OVERLAP,
        save_coords=SAVE_COORDINATES,
        patch_format=patch_format,
        jpeg_quality=jpeg_quality
    )
    
    if workers <= 1:
        # Sequential mode (for debugging)
        for i, tiff in enumerate(tiff_files, 1):
            result = worker_fn(tiff)
            pct = 100 * i / total_wsis
            
            if result['status'] == 'ok':
                processed += 1
                total_patches += result['patches']
                print(f"  [{i}/{total_wsis} = {pct:.1f}%] {result['wsi']}: {result['patches']} patches")
            else:
                failed += 1
                print(f"  [{i}/{total_wsis} = {pct:.1f}%] {result['wsi']}: FAILED - {result.get('error', 'unknown')}")
    else:
        # Multiprocessing mode
        print(f"  Starting parallel processing with {workers} workers...")
        
        with mp.Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_fn, tiff_files), 1):
                pct = 100 * i / total_wsis
                
                if result['status'] == 'ok':
                    processed += 1
                    total_patches += result['patches']
                    print(f"  [{i}/{total_wsis} = {pct:.1f}%] {result['wsi']}: {result['patches']} patches")
                else:
                    failed += 1
                    print(f"  [{i}/{total_wsis} = {pct:.1f}%] {result['wsi']}: FAILED - {result.get('error', 'unknown')}")
    
    total_elapsed = time.time() - start_time
    
    return {
        'wsi_count': total_wsis,
        'patch_count': total_patches,
        'processed': processed,
        'failed': failed,
        'status': 'complete',
        'elapsed_hours': total_elapsed / 3600
    }


def main():
    parser = argparse.ArgumentParser(description='source C WSI Patch Extraction Pipeline')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Process specific dataset (e.g., source C-skin-b1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    parser.add_argument('--patch-size', type=int, default=CONFIG_PATCH_SIZE,
                       help=f'Patch size in pixels (default: {CONFIG_PATCH_SIZE})')
    parser.add_argument('--min-fg-ratio', type=float, default=CONFIG_MIN_FG,
                       help=f'Minimum foreground ratio (default: {CONFIG_MIN_FG})')
    parser.add_argument('--output', type=str, default=CONFIG_OUTPUT_BASE,
                       help=f'Output directory (default: {CONFIG_OUTPUT_BASE})')
    parser.add_argument('--workers', type=int, default=CONFIG_MAX_WORKERS,
                       help=f'Number of parallel workers (default: {CONFIG_MAX_WORKERS}, use 1 for sequential)')
    parser.add_argument('--format', type=str, default=CONFIG_PATCH_FORMAT,
                       choices=['jpeg', 'png'],
                       help=f'Output image format (default: {CONFIG_PATCH_FORMAT})')
    parser.add_argument('--jpeg-quality', type=int, default=CONFIG_JPEG_QUALITY,
                       help=f'JPEG quality 1-100 (default: {CONFIG_JPEG_QUALITY})')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("source C WSI Patch Extraction Pipeline")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source: {SOURCE_C_BASE}")
    print(f"Output: {CONFIG_OUTPUT_BASE}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Min foreground ratio: {args.min_fg_ratio}")
    print(f"Format: {args.format} (JPEG quality: {args.jpeg_quality})" if args.format == 'jpeg' else f"Format: {args.format}")
    print(f"Workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)
    
    # Select datasets
    datasets_to_process = DATASETS
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(DATASETS)}")
            sys.exit(1)
        datasets_to_process = [args.dataset]
    
    # Skip metadata (no images)
    datasets_to_process = [d for d in datasets_to_process if d != 'source C-metadata']
    
    # Load case selection if available
    case_selection = load_case_selection()
    if case_selection:
        print(f"\n  Case selection file loaded: {CASE_SELECTION_FILE}")
        for ds, cases in case_selection.items():
            print(f"    {ds}: {len(cases)} cases selected")
    else:
        print("\n  No case selection file - processing all cases")
    
    overall_start = time.time()
    results = {}
    
    for i, dataset_name in enumerate(datasets_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Processing Dataset [{i}/{len(datasets_to_process)}]: {dataset_name}")
        print(f"{'='*80}")
        
        result = process_dataset(
            dataset_name, args.dry_run, case_selection,
            workers=args.workers,
            patch_size=args.patch_size,
            min_fg_ratio=args.min_fg_ratio,
            patch_format=args.format,
            jpeg_quality=args.jpeg_quality
        )
        results[dataset_name] = result
        
        if result['status'] == 'complete':
            print(f"\n  Summary: {result['processed']} processed, {result['failed']} failed, {result['patch_count']:,} patches")
            print(f"  Time: {result.get('elapsed_hours', 0):.2f} hours")
        elif result['status'] == 'dry_run':
            print(f"\n  [DRY RUN] {result['wsi_count']} WSIs would be processed")
    
    # Overall summary
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_wsis = 0
    total_patches = 0
    total_failed = 0
    
    print(f"{'Dataset':<30} {'WSIs':>8} {'Patches':>12} {'Failed':>8} {'Time(h)':>8} {'Status':<12}")
    print("-" * 88)
    
    for ds, res in results.items():
        wsis = res.get('wsi_count', 0)
        patches = res.get('patch_count', 0)
        failed = res.get('failed', 0)
        status = res.get('status', 'unknown')
        hrs = res.get('elapsed_hours', 0)
        
        total_wsis += wsis
        total_patches += patches
        total_failed += failed
        
        print(f"{ds:<30} {wsis:>8} {patches:>12,} {failed:>8} {hrs:>8.2f} {status:<12}")
    
    print("-" * 88)
    print(f"{'TOTAL':<30} {total_wsis:>8} {total_patches:>12,} {total_failed:>8}")
    print(f"{'='*80}")
    print(f"Total time: {overall_elapsed/3600:.2f} hours")
    print(f"Workers used: {args.workers}")
    
    if not args.dry_run:
        fmt_label = "JPEG" if args.format == 'jpeg' else "PNG"
        print(f"\nOutput saved to: {CONFIG_OUTPUT_BASE}")
        print(f"Each WSI has its own directory with:")
        print(f"  - {fmt_label} patch files (named with coordinates)")
        print(f"  - _patches.csv (coordinate metadata)")


if __name__ == "__main__":
    main()
