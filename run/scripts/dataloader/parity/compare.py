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

"""Compare two JSON parity dumps field-by-field.

Run:
    python -m nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.compare \
        --a /tmp/parity/tao.json --b /tmp/parity/evfm.json

Walks the JSON structure recursively.  For large lists, reports a single
pass/fail with the first mismatch.  Float comparison uses math.isclose
with the same default tolerances as torch.allclose (rtol=1e-5, atol=1e-8).

Exit code 0 = all match, 1 = mismatches found.
"""

import argparse
import json
import logging
import math
import sys

logger = logging.getLogger(__name__)

# Same defaults as torch.allclose for float32 comparisons.
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8


class ParityResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def report(self, path, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        msg = f"  [{tag}] {path}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


def _compare(a, b, path, result, rtol, atol):
    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = sorted(set(list(a.keys()) + list(b.keys())))
        for k in all_keys:
            child = f"{path}.{k}" if path else k
            if k not in a:
                result.report(child, False, "missing in A")
            elif k not in b:
                result.report(child, False, "missing in B")
            else:
                _compare(a[k], b[k], child, result, rtol, atol)

    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            result.report(path, False, f"len {len(a)} vs {len(b)}")
            return
        if len(a) == 0:
            result.report(path, True, "both empty")
            return

        if len(a) > 20:
            mismatches = 0
            first_mm = None
            for i, (va, vb) in enumerate(zip(a, b)):
                if isinstance(va, float) and isinstance(vb, float):
                    if not math.isclose(va, vb, rel_tol=rtol, abs_tol=atol):
                        mismatches += 1
                        if first_mm is None:
                            first_mm = (i, va, vb)
                elif va != vb:
                    mismatches += 1
                    if first_mm is None:
                        first_mm = (i, va, vb)
            ok = mismatches == 0
            detail = f"len={len(a)}"
            if not ok:
                detail += (
                    f", {mismatches} mismatches, "
                    f"first@[{first_mm[0]}]: {first_mm[1]} vs {first_mm[2]}"
                )
            result.report(path, ok, detail)
        else:
            for i, (va, vb) in enumerate(zip(a, b)):
                _compare(va, vb, f"{path}[{i}]", result, rtol, atol)

    elif isinstance(a, float) and isinstance(b, float):
        ok = math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        detail = "" if ok else f"{a} vs {b}, diff={abs(a - b):.2e}"
        result.report(path, ok, detail)

    elif isinstance(a, (int, str, bool, type(None))) and type(a) == type(b):
        ok = a == b
        detail = "" if ok else f"{a!r} vs {b!r}"
        result.report(path, ok, detail)

    else:
        ok = a == b
        detail = (
            "" if ok
            else f"type {type(a).__name__} vs {type(b).__name__}: "
                 f"{str(a)[:80]} vs {str(b)[:80]}"
        )
        result.report(path, ok, detail)


def compare(a, b, components=None, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Compare two dump dicts.  Returns a ParityResult."""
    if components:
        a = {k: v for k, v in a.items() if k in components}
        b = {k: v for k, v in b.items() if k in components}

    result = ParityResult()
    _compare(a, b, "", result, rtol, atol)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Compare two parity JSON dumps")
    parser.add_argument("--a", required=True, help="First JSON dump (e.g. TAO)")
    parser.add_argument("--b", required=True, help="Second JSON dump (e.g. reference)")
    parser.add_argument(
        "--components", nargs="*", default=None,
        help="Compare only these top-level keys (default: all)",
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                        help=f"Absolute tolerance (default: {DEFAULT_ATOL})")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=f"Relative tolerance (default: {DEFAULT_RTOL})")
    args = parser.parse_args()

    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)

    print("=" * 60)
    print("Parity Comparison")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}")
    print(f"  rtol: {args.rtol},  atol: {args.atol}")
    if args.components:
        print(f"  components: {args.components}")
    print("=" * 60)

    result = compare(a, b, args.components, rtol=args.rtol, atol=args.atol)

    print("\n" + "=" * 60)
    print(f"Results: {result.passed} passed, {result.failed} failed")
    print("=" * 60)
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()
