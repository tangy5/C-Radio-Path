#!/usr/bin/env python3
"""
Validate source C webdataset integrity — checks for corrupted samples that would
crash training (truncated tars, bad JPEGs, broken JSON, sha256 mismatches).

Usage:
    python validate_source_c_webdataset.py                          # Full validation, all shards
    python validate_source_c_webdataset.py --quick                  # Tar integrity + JPEG magic only (fast)
    python validate_source_c_webdataset.py --shards 1-10            # Validate specific shard range
    python validate_source_c_webdataset.py --report report.json     # Save detailed report
"""

import os
import sys
import json
import glob
import time
import struct
import hashlib
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
WEBDATASET_DIR = "<PATHOLOGY_DATASETS_DIR>/source_c_webdataset"

JPEG_MAGIC = b'\xff\xd8'


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def validate_jpeg_header(data: bytes) -> bool:
    """Check JPEG magic bytes without full decode."""
    if len(data) < 4:
        return False
    return data[:2] == JPEG_MAGIC

def validate_jpeg_full(data: bytes) -> tuple[bool, str]:
    """Full JPEG decode via PIL (slower but catches truncated images)."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        img.verify()  # verify integrity without full decode
        return True, f"OK ({img.size[0]}x{img.size[1]})"
    except Exception as e:
        return False, f"PIL error: {e}"


def validate_shard(shard_path, quick=False, sha256_check=False, deep_jpeg=False):
    """Validate a single shard. Returns dict with error counts and details."""
    stats = {
        "shard": os.path.basename(shard_path),
        "total_samples": 0,
        "ok": 0,
        "errors": [],
    }

    # 1. Tar integrity
    try:
        tf = tarfile.open(shard_path, 'r')
        members = tf.getmembers()
    except Exception as e:
        stats["errors"].append(f"TAR_CORRUPT: cannot open tar: {e}")
        return stats

    # Group by key (number prefix)
    samples = defaultdict(dict)
    for m in members:
        name = m.name
        base, ext = os.path.splitext(name)
        key = base  # e.g. "000000000"
        if ext == ".jpg":
            samples[key]["jpg_member"] = m
        elif ext == ".json":
            samples[key]["json_member"] = m

    stats["total_samples"] = len(samples)

    for key, sample in sorted(samples.items()):
        errors_this = []

        # 2. Pair check
        jpg_member = sample.get("jpg_member")
        json_member = sample.get("json_member")

        if not jpg_member:
            errors_this.append("MISSING_JPG")
        if not json_member:
            errors_this.append("MISSING_JSON")

        if not jpg_member or not json_member:
            stats["errors"].append(f"{key}: {'; '.join(errors_this)}")
            continue

        # 3. Read JPEG bytes
        try:
            jpg_f = tf.extractfile(jpg_member)
            jpg_data = jpg_f.read()
        except Exception as e:
            stats["errors"].append(f"{key}: JPEG_READ_ERROR: {e}")
            continue

        if len(jpg_data) == 0:
            stats["errors"].append(f"{key}: JPEG_EMPTY")
            continue

        # 4. JPEG header check (always)
        if not validate_jpeg_header(jpg_data):
            stats["errors"].append(f"{key}: BAD_JPEG_HEADER: starts with {jpg_data[:4].hex()}")
            continue

        # 5. Deep JPEG decode (optional, slow)
        if deep_jpeg:
            ok, msg = validate_jpeg_full(jpg_data)
            if not ok:
                stats["errors"].append(f"{key}: TRUNCATED_JPEG: {msg}")
                continue

        # 6. Read JSON bytes
        try:
            json_f = tf.extractfile(json_member)
            json_data = json_f.read()
        except Exception as e:
            stats["errors"].append(f"{key}: JSON_READ_ERROR: {e}")
            continue

        if len(json_data) == 0:
            stats["errors"].append(f"{key}: JSON_EMPTY")
            continue

        # 7. JSON parse
        try:
            meta = json.loads(json_data)
        except json.JSONDecodeError as e:
            stats["errors"].append(f"{key}: JSON_PARSE_ERROR: {e}")
            continue

        # 8. SHA256 check (optional, slow for large shards)
        if sha256_check:
            expected = meta.get("sha256")
            if not expected:
                stats["errors"].append(f"{key}: JSON_MISSING_SHA256")
                continue
            actual = compute_sha256(jpg_data)
            if actual != expected:
                stats["errors"].append(f"{key}: SHA256_MISMATCH: expected {expected[:16]}..., got {actual[:16]}...")
                continue

        stats["ok"] += 1

    tf.close()

    if not quick:
        # Verify no orphaned files (unmatched keys)
        stats["files_in_tar"] = len(members)
        stats["jpg_files"] = sum(1 for m in members if m.name.endswith('.jpg'))
        stats["json_files"] = sum(1 for m in members if m.name.endswith('.json'))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate source C webdataset")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: tar integrity + JPEG header only (no json/sha256)")
    parser.add_argument("--sha256", action="store_true",
                        help="Verify SHA256 hashes match image bytes (slow)")
    parser.add_argument("--deep-jpeg", action="store_true",
                        help="Full PIL-based JPEG decode to catch truncation (very slow)")
    parser.add_argument("--shards", type=str, default=None,
                        help="Shard range, e.g. '1-10' or 'all' (default: all)")
    parser.add_argument("--report", type=str, default=None,
                        help="Save detailed JSON report to file")
    parser.add_argument("--output", type=str, default=WEBDATASET_DIR,
                        help=f"Webdataset directory (default: {WEBDATASET_DIR})")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N shards to check (0 = all)")
    args = parser.parse_args()

    print("=" * 70)
    print("source C Webdataset Quality Check")
    print("=" * 70)
    print(f"Directory: {args.output}")
    print(f"Mode: {'Quick (header only)' if args.quick else 'Full (tar + jpeg + json)'}")
    print(f"SHA256 check: {'ON' if args.sha256 else 'OFF'}")
    print(f"Deep JPEG decode: {'ON' if args.deep_jpeg else 'OFF'}")
    print("=" * 70)

    # Find all shards
    all_shards = sorted(glob.glob(os.path.join(args.output, "**", "*.tar"), recursive=True))
    if not all_shards:
        print(f"ERROR: No tar files found in {args.output}")
        sys.exit(1)

    # Filter by range
    if args.shards:
        parts = args.shards.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            all_shards = [s for s in all_shards
                          if start <= int(os.path.basename(s).split("_")[1].split(".")[0]) <= end]
        else:
            print(f"Invalid --shards format. Use '1-10'.")
            sys.exit(1)

    # Sample if requested
    if args.sample > 0 and args.sample < len(all_shards):
        import random
        random.seed(42)
        all_shards = sorted(random.sample(all_shards, args.sample))

    print(f"\nChecking {len(all_shards)} shards...\n")

    t0 = time.time()
    all_results = []
    total_ok = 0
    total_errors = 0
    total_samples = 0
    corrupt_tars = 0

    for i, shard_path in enumerate(all_shards):
        result = validate_shard(shard_path, quick=args.quick,
                                sha256_check=args.sha256,
                                deep_jpeg=args.deep_jpeg)
        all_results.append(result)

        total_samples += result["total_samples"]
        total_ok += result["ok"]
        num_errors = len(result["errors"])
        total_errors += num_errors
        if any(e.startswith("TAR_CORRUPT") for e in result["errors"]):
            corrupt_tars += 1

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = len(all_shards) - i - 1
        eta = remaining / rate if rate > 0 else 0

        status = "CORRUPT" if corrupt_tars and any(e.startswith("TAR_CORRUPT")
                  for e in result["errors"]) else ("WARN" if num_errors > 0 else "OK")
        print(f"  [{i+1:4d}/{len(all_shards)}] {status:>7s}  "
              f"{result['shard']:>20s}  "
              f"samples={result['ok']}/{result['total_samples']}  "
              f"errors={num_errors}  "
              f"| {rate:.1f} shard/s | ETA {eta:.0f}s",
              flush=True)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Shards checked:     {len(all_shards):,}")
    print(f"  Total samples:      {total_samples:,}")
    print(f"  OK samples:         {total_ok:,} ({100*total_ok/total_samples:.2f}%)"
          if total_samples > 0 else "  OK samples:         0")
    print(f"  Error samples:      {total_errors:,}")
    print(f"  Corrupt tar files:  {corrupt_tars}")
    print(f"  Time:               {elapsed:.1f}s ({len(all_shards)/elapsed:.1f} shard/s)" if elapsed > 0 else "")
    print(f"{'='*70}")

    # Warnings
    issues_by_type = defaultdict(int)
    for r in all_results:
        for e in r["errors"]:
            etype = e.split(":", 1)[0].strip() if ":" in e else e
            issues_by_type[etype] += 1

    if issues_by_type:
        print(f"\nISSUE BREAKDOWN:")
        for etype, count in sorted(issues_by_type.items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count}")
    else:
        print(f"\nAll samples passed. No issues found.")

    # Save report
    if args.report:
        report = {
            "directory": args.output,
            "shards_checked": len(all_shards),
            "total_samples": total_samples,
            "ok_samples": total_ok,
            "error_samples": total_errors,
            "corrupt_tars": corrupt_tars,
            "time_seconds": elapsed,
            "issues_by_type": dict(issues_by_type),
            "shard_details": all_results,
        }
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.report}")

    # Exit code
    if corrupt_tars > 0:
        sys.exit(2)  # Corruption detected
    elif total_errors > 0:
        sys.exit(1)  # Warnings
    else:
        sys.exit(0)  # Clean


if __name__ == "__main__":
    main()
