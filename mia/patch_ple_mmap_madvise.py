#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""patch_ple_mmap_madvise.py -- MADV_RANDOM for the PLE mmap accesses.

Origin of the idea (research 10.09.2026, MiaAI-Lab measurement): PLE n-gram
accesses are near-random small row accesses (n-gram lookup). Without
MADV_RANDOM the kernel pulls the read-ahead window (128 KiB here) into the page
cache on every page fault; on the Spark the page cache is the same physical
LPDDR5X pool as GPU memory. MiaAI measured 1366 KiB of SSD read per decoded
token without vs. 57 KiB with MADV_RANDOM -- roughly 2 GiB of page cache
completely unused.

Verified finding (10.09.2026): blazux/qwen3.8-Flash-DGX @ 656cdb0 (our base
image) calls madvise/MADV_RANDOM NOWHERE (grep over the entire repo); the
MiaAI stack uses its own packed PLE table instead of mmap and therefore does
not cover our path. This patch closes that gap.

How it works: after MmapPleTable.__init__ every np.memmap view that was created
is marked MADV_RANDOM. A pure behavioural optimum for random accesses; a wrong
prediction costs one more fault, never a correct result.

Flow (identical to patch_mtp_draft_vocab.py): build_image_mia extracts the
original vllm_ple_mmap.py from VLLM_IMAGE into ./vllm_ple_mmap_patched.py.orig;
this script writes ./vllm_ple_mmap_patched.py. The patch verifies each anchor
exactly once and aborts on drift with exit code 1 (fail-safe, no silent
switching to a fallback).

Switch inside the container: VLLM_PLE_MMAP_MADV_RANDOM (default 1 = on, "0" =
off). The patched module lives in the mia layer; k3 keeps running on
VLLM_IMAGE and stays untouched (smallest intrusion into a running setup).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "vllm_ple_mmap_patched.py.orig")
OUT = os.path.join(HERE, "vllm_ple_mmap_patched.py")

# Exactly the docstring block start from vllm_ple_mmap.py @ 656cdb0 (en dash
# U+2013 in the original). The patch aborts if this line changes.
DOCSTRING_ANCHOR = (
    '"""vllm_ple_mmap \u2014 serve the Qwen3.8-Flash-Next N-gram (PLE) '
    "table from NVMe via mmap."
)

MADVISE_BLOCK = '''

# --- qwen38-flash-next-vllm-serve: MADV_RANDOM for PLE accesses --------------
# MiaAI-Lab measurement: PLE lookups are random accesses; without MADV_RANDOM
# the read-ahead window fills the page cache with data that is never
# read (~1366 KiB vs. 57 KiB SSD read per decoded token; up to ~2 GiB of
# page cache unused). On the Spark the page cache is the same physical pool
# as GPU memory. Switch: VLLM_PLE_MMAP_MADV_RANDOM (default: on; "0" =
# off). Handle resolution empirically verified on the numpy of the image
# (2.2.6, stripped from qwen38-flash-dgx:656cdb0): np.memmap keeps the
# mmap object under `._mmap`; a public `.mmap` does not exist there.
# Accept only a handle that really can madvise() -- an attribute with a merely
# similar name would otherwise silently do nothing (2 GiB page-cache lever
# without a single error message).
def _ple_madvise_enabled() -> bool:
    return os.environ.get("VLLM_PLE_MMAP_MADV_RANDOM", "1").strip() != "0"


def _ple_madvise_random_after_init(self, *args, **kwargs):
    _ple_madvise_random_orig_init(self, *args, **kwargs)
    if not _ple_madvise_enabled():
        logger.info("PLE mmap: MADV_RANDOM disabled (VLLM_PLE_MMAP_MADV_RANDOM=0)")
        return
    import mmap as _mmap_mod

    def _madvise_handle(_obj):
        # np.memmap keeps the mmap object in `._mmap`. Putting _mmap first and
        # the madvise check together make the patch robust against NumPy
        # versions that populate `.mmap` differently.
        for _name in ("_mmap", "mmap"):
            _cand = getattr(_obj, _name, None)
            if callable(getattr(_cand, "madvise", None)):
                return _cand
        return None

    n = 0
    for _mm in getattr(self, "mm", []):
        _backing = _madvise_handle(_mm)
        if _backing is None:
            continue
        try:
            _backing.madvise(_mmap_mod.MADV_RANDOM)
            n += 1
        except Exception as _exc:  # pragma: no cover - safety net, never fatal
            logger.warning("PLE mmap: MADV_RANDOM failed (%s)", _exc)
            return
    if n:
        logger.info(
            "PLE mmap: MADV_RANDOM set on %d shard views "
            "(no read-ahead page cache for n-gram lookups).", n)
    else:
        logger.warning(
            "PLE mmap: no mmap handle found on the np.memmap views "
            "(NumPy API change?) -- MADV_RANDOM NOT set.")


_ple_madvise_random_orig_init = MmapPleTable.__init__
MmapPleTable.__init__ = _ple_madvise_random_after_init
# --- end MADV_RANDOM patch ---------------------------------------------------
'''


def main() -> int:
    if not os.path.exists(ORIG):
        print(f"!! original missing: {ORIG}", file=sys.stderr)
        print("   (build-image-mia extracts it from VLLM_IMAGE).", file=sys.stderr)
        return 1
    src = open(ORIG, encoding="utf-8").read()

    if "MADV_RANDOM" in src:
        print("!! vllm_ple_mmap.py already contains MADV_RANDOM -- "
              "upstream drift, check the patch against the new source.",
              file=sys.stderr)
        return 1
    if src.count(DOCSTRING_ANCHOR) != 1:
        print(f"!! docstring anchor found {src.count(DOCSTRING_ANCHOR)}x "
              "(expected 1x) -- vllm_ple_mmap.py changed against 656cdb0; "
              "do NOT apply the patch silently.", file=sys.stderr)
        return 1
    if ("class MmapPleTable" not in src or "def apply(cls: type) -> None:" not in src
            or "logger = " not in src):
        # Structural check: the class, the apply signature and the module logger
        # must exist (the block references them); missing anchors = upstream drift.
        print("!! structure anchors (class MmapPleTable / def apply / logger) not "
              "found -- upstream drift.", file=sys.stderr)
        return 1

    patched = src.rstrip("\n") + "\n" + MADVISE_BLOCK
    ast.parse(patched)
    if patched.count("MADV_RANDOM") < 4:
        print("!! sanity: MADV_RANDOM appears surprisingly rarely in the patched "
              "module.", file=sys.stderr)
        return 1
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(patched)
    print(f"OK: MADV_RANDOM patch applied to MmapPleTable -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
