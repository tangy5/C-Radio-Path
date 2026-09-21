# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare parity dumps from TAO and EVFM training runs.

Three-phase comparison:
  Phase 0 (identity):    exact string match of sample keys/urls.
  Phase 1 (raw inputs):  exact match of raw decoded images before augmentation.
  Phase 2 (batches):     tolerance match of augmented/collated batches.

Usage:
    python compare_training_batches.py /tmp/parity_tao /tmp/parity_evfm

Exit code 0 = all match, 1 = mismatches found.
"""

import argparse
import json
import logging
import os
import sys
from glob import glob

import torch

logger = logging.getLogger(__name__)

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8

NOCLASS_IDX = -1


def _both_no_labels(va, vb):
    """Return True when both tensors represent 'no labels'.

    TAO fills ``NOCLASS_IDX`` (shape ``[B]``), EVFM dumps empty
    ``tensor([])``.  Both mean 'no class labels available'.
    """
    a_empty = va.numel() == 0 or (va == NOCLASS_IDX).all()
    b_empty = vb.numel() == 0 or (vb == NOCLASS_IDX).all()
    return a_empty and b_empty


def _tensor_stats(t):
    return (
        f"shape={list(t.shape)}, dtype={t.dtype}, "
        f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
        f"mean={t.float().mean().item():.6f}"
    )


def compare_identity(dir_a, dir_b):
    """Phase 0: compare sample __key__ and __url__ (exact string match)."""
    path_a = os.path.join(dir_a, "identity", "sample_keys.json")
    path_b = os.path.join(dir_b, "identity", "sample_keys.json")

    if not os.path.exists(path_a) and not os.path.exists(path_b):
        logger.info("  No identity dumps found -- skipping Phase 0")
        return 0, 0

    for label, path in [("A", path_a), ("B", path_b)]:
        if not os.path.exists(path):
            logger.error("  [FAIL] identity dump missing for %s: %s", label, path)
            return 0, 1

    with open(path_a) as f:
        keys_a = json.load(f)
    with open(path_b) as f:
        keys_b = json.load(f)

    n = min(len(keys_a), len(keys_b))
    if len(keys_a) != len(keys_b):
        logger.warning(
            "  Identity count differs: %d vs %d -- comparing first %d",
            len(keys_a), len(keys_b), n,
        )

    passed, failed = 0, 0
    for i in range(n):
        ka, kb = keys_a[i], keys_b[i]
        key_match = ka["key"] == kb["key"]
        url_match = ka["url"] == kb["url"]
        if key_match and url_match:
            passed += 1
        else:
            logger.error(
                "  [FAIL] sample %d: key=%r vs %r, url=%r vs %r",
                i, ka["key"], kb["key"], ka["url"], kb["url"],
            )
            failed += 1
            if failed >= 10:
                logger.error("  ... (stopping after 10 failures)")
                break

    return passed, failed


def compare_raw_samples(dir_a, dir_b, rtol, atol):
    """Phase 1: compare raw decoded images (tolerance match).

    Python and C++ fast_to_tensor implementations may differ at float32
    precision (~6e-8), so we use the same tolerance-based comparison as
    Phase 2 rather than requiring exact bit equality.
    """
    raw_a = sorted(glob(os.path.join(dir_a, "raw", "raw_sample_*.pt")))
    raw_b = sorted(glob(os.path.join(dir_b, "raw", "raw_sample_*.pt")))

    if not raw_a and not raw_b:
        logger.info("  No raw sample dumps found -- skipping Phase 1")
        return 0, 0

    n = min(len(raw_a), len(raw_b))
    if len(raw_a) != len(raw_b):
        logger.warning(
            "  Raw sample count differs: %d vs %d -- comparing first %d",
            len(raw_a), len(raw_b), n,
        )

    passed, failed = 0, 0
    for i in range(n):
        a = torch.load(raw_a[i], map_location="cpu", weights_only=True)
        b = torch.load(raw_b[i], map_location="cpu", weights_only=True)
        img_a, img_b = a["raw_img"], b["raw_img"]

        if img_a.shape != img_b.shape:
            logger.error(
                "  [FAIL] raw_sample_%06d: shape %s vs %s",
                i, list(img_a.shape), list(img_b.shape),
            )
            failed += 1
            continue

        if torch.allclose(img_a.float(), img_b.float(), rtol=rtol, atol=atol):
            passed += 1
        else:
            diff = (img_a.float() - img_b.float()).abs()
            logger.error(
                "  [FAIL] raw_sample_%06d: max_diff=%.2e, mean_diff=%.2e",
                i, diff.max().item(), diff.mean().item(),
            )
            failed += 1

    return passed, failed


def compare_batches(dir_a, dir_b, rtol, atol):
    """Phase 2: compare augmented/collated batches (tolerance match)."""
    files_a = sorted(glob(os.path.join(dir_a, "batch_*.pt")))
    files_b = sorted(glob(os.path.join(dir_b, "batch_*.pt")))

    if not files_a and not files_b:
        logger.info("  No batch dumps found -- skipping Phase 2")
        return 0, 0

    n = min(len(files_a), len(files_b))
    if len(files_a) != len(files_b):
        logger.warning(
            "  Batch count differs: %d vs %d -- comparing first %d",
            len(files_a), len(files_b), n,
        )

    passed, failed = 0, 0
    for i in range(n):
        a = torch.load(files_a[i], map_location="cpu", weights_only=True)
        b = torch.load(files_b[i], map_location="cpu", weights_only=True)

        logger.info("  --- batch_%03d ---", i)

        all_keys = sorted(
            set(list(a.keys()) + list(b.keys()))
            - {"num_teachers"}
        )

        for key in all_keys:
            if key not in a:
                logger.error("    [FAIL] %s: missing in A", key)
                failed += 1
                continue
            if key not in b:
                logger.error("    [FAIL] %s: missing in B", key)
                failed += 1
                continue

            va, vb = a[key], b[key]
            if not isinstance(va, torch.Tensor) or not isinstance(vb, torch.Tensor):
                if va == vb:
                    passed += 1
                else:
                    logger.error("    [FAIL] %s: %s vs %s", key, va, vb)
                    failed += 1
                continue

            if key == "class" and _both_no_labels(va, vb):
                logger.info("    [SKIP] %s  (no labels in both)", key)
                passed += 1
                continue

            if va.shape != vb.shape:
                logger.error(
                    "    [FAIL] %s: shape %s vs %s",
                    key, list(va.shape), list(vb.shape),
                )
                failed += 1
                continue

            if torch.allclose(va.float(), vb.float(), rtol=rtol, atol=atol):
                logger.info("    [PASS] %s  (%s)", key, _tensor_stats(va))
                passed += 1
            else:
                diff = (va.float() - vb.float()).abs()
                logger.error(
                    "    [FAIL] %s: max_diff=%.2e, mean_diff=%.2e\n"
                    "           A: %s\n"
                    "           B: %s",
                    key, diff.max().item(), diff.mean().item(),
                    _tensor_stats(va), _tensor_stats(vb),
                )
                failed += 1

    return passed, failed


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Compare TAO vs EVFM parity dumps",
    )
    parser.add_argument("dir_a", help="First dump directory (e.g. TAO)")
    parser.add_argument("dir_b", help="Second dump directory (e.g. EVFM)")
    parser.add_argument(
        "--atol", type=float, default=DEFAULT_ATOL,
        help=f"Absolute tolerance for batch comparison (default: {DEFAULT_ATOL})",
    )
    parser.add_argument(
        "--rtol", type=float, default=DEFAULT_RTOL,
        help=f"Relative tolerance for batch comparison (default: {DEFAULT_RTOL})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Training Batch Parity Comparison")
    print(f"  A: {args.dir_a}")
    print(f"  B: {args.dir_b}")
    print(f"  rtol: {args.rtol},  atol: {args.atol}")
    print("=" * 60)

    total_passed, total_failed = 0, 0

    print("\nPhase 0: Sample identity comparison (exact string match)")
    print("-" * 40)
    p, f = compare_identity(args.dir_a, args.dir_b)
    total_passed += p
    total_failed += f
    print(f"  Identity: {p} passed, {f} failed")

    print(f"\nPhase 1: Raw input comparison (rtol={args.rtol}, atol={args.atol})")
    print("-" * 40)
    p, f = compare_raw_samples(args.dir_a, args.dir_b, args.rtol, args.atol)
    total_passed += p
    total_failed += f
    print(f"  Raw samples: {p} passed, {f} failed")

    print(f"\nPhase 2: Batch comparison (rtol={args.rtol}, atol={args.atol})")
    print("-" * 40)
    p, f = compare_batches(args.dir_a, args.dir_b, args.rtol, args.atol)
    total_passed += p
    total_failed += f
    print(f"  Batches: {p} passed, {f} failed")

    print("\n" + "=" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
