#!/usr/bin/env python3
"""Remove bad quality patches identified by check_patch_quality.py.

Reads bad_files_*.csv from the logs directory, deduplicates filepaths,
deletes bad patches, and logs every deletion.

Usage:
    python remove_bad_patches.py                    # Remove all bad patches
    python remove_bad_patches.py --dry-run          # Show what would be deleted
    python remove_bad_patches.py --dataset source C-breast  # Single dataset
"""

import os
import sys
import csv
import glob
import argparse
from pathlib import Path

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def load_bad_files(csv_pattern):
    """Load unique filepaths from bad_files CSVs."""
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        print(f"No bad_files CSVs found matching: {csv_pattern}")
        return []

    bad_files = set()
    for csv_path in csv_files:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row.get('filepath', '').strip()
                if fp:
                    bad_files.add(fp)

    return sorted(bad_files)


def main():
    parser = argparse.ArgumentParser(description='Remove bad quality source C patches')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Only remove bad patches from a specific dataset')
    args = parser.parse_args()

    if args.dataset:
        csv_pattern = os.path.join(LOGS_DIR, f'bad_files_{args.dataset}.csv')
    else:
        csv_pattern = os.path.join(LOGS_DIR, 'bad_files_*.csv')

    bad_files = load_bad_files(csv_pattern)
    if not bad_files:
        print("No bad files found to remove.")
        return

    print(f"Found {len(bad_files):,} unique bad files to remove.")
    print(f"Dry run: {'YES' if args.dry_run else 'NO'}")

    removed = 0
    failed = 0
    total_bytes = 0
    not_found = 0

    for fp in bad_files:
        if os.path.exists(fp):
            fsize = os.path.getsize(fp)
            total_bytes += fsize
            if not args.dry_run:
                try:
                    os.remove(fp)
                    removed += 1
                except OSError as e:
                    print(f"  FAILED: {fp}: {e}")
                    failed += 1
            else:
                removed += 1
        else:
            not_found += 1
            # Already deleted or path issue
            pass

    if not_found > 0:
        print(f"\n  Note: {not_found} files already missing (already removed or path issue)")

    print(f"\n{'='*60}")
    print(f"{'DRY RUN - would remove' if args.dry_run else 'Removed'} {removed:,} bad patches")
    if failed > 0:
        print(f"  Failed to remove: {failed}")
    print(f"  Total space freed: {total_bytes / (1024**3):.2f} GB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
