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

"""Parity testing framework.

Provides a component registry and CLI for dumping deterministic outputs
from both the TAO dataloader and a reference implementation.  Two dumps
can then be compared field-by-field using ``compare.py``.

To add a new component, decorate a function with ``@component("name")``
in the appropriate ``tao/`` or ``evfm/`` sub-module.  The function must
return a JSON-serialisable dict.
"""

import argparse
import json
import logging
import os
import sys
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[], dict]] = {}


def component(name: str):
    """Decorator: register a dump function under *name*."""
    def decorator(fn: Callable[[], dict]):
        if name in _REGISTRY:
            raise ValueError(f"Duplicate parity component: {name!r}")
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_registry() -> Dict[str, Callable[[], dict]]:
    return _REGISTRY


def reset_registry():
    """Clear all registered components (useful between tao/evfm runs)."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Dump CLI (called by tao/__main__.py and evfm/__main__.py)
# ---------------------------------------------------------------------------

def dump_main():
    """Run registered dump functions and write JSON output."""
    parser = argparse.ArgumentParser(
        description="Dump component outputs for parity comparison",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON dump")
    parser.add_argument(
        "--components", nargs="*", default=None,
        help="Run only these components (default: all registered)",
    )
    parser.add_argument("--list", action="store_true", help="List registered components and exit")
    args = parser.parse_args()

    reg = get_registry()

    if args.list:
        for name in sorted(reg):
            print(f"  {name}")
        sys.exit(0)

    names = args.components if args.components else sorted(reg)
    unknown = set(names) - set(reg)
    if unknown:
        logger.error("Unknown components: %s.  Available: %s", unknown, sorted(reg))
        sys.exit(1)

    dump = {}
    for name in names:
        logger.info("dumping %s ...", name)
        dump[name] = reg[name]()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dump, f, indent=2, sort_keys=True)

    logger.info("Wrote %s  (%d bytes, %d components)", args.output, os.path.getsize(args.output), len(names))
