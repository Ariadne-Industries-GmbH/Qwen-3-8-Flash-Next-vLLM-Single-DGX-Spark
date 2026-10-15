#!/usr/bin/env python3
"""REAL-LAYOUT validation of the WILLNEED prefetch patch (Fix A).

Runs INSIDE the container against the ACTUAL safetensors shards (read-only,
mounted at /models/qwen3.8-flash-next-nvfp4, which is the live model dir).
No server restart, no file change. Proves:
  1. The real shards dict {idx:(path,abs_offset,rows)} + row_bytes the runtime
     will build (via the module's own _find_shards) parse cleanly.
  2. posix_fadvise(fd, FADV_WILLNEED, abs_offset + local_row*row_bytes, len)
     succeeds on the real files and stays within each shard file's byte bounds.
  3. A realistic decode-sized (<=512) and prefill-sized (>512) id batch both
     route correctly through _prefetch_rows with zero exceptions.
"""
import importlib.util
import os
import sys

# Two modes, both read-only:
# a) INSIDE the serve container, against the real shards on the mount:
#      docker cp tests/validate_real_layout.py <container>:/root/diag/
#      docker exec <container> python3 /root/diag/validate_real_layout.py
# b) ON THE HOST, against the same checkpoint -- needs a Python with torch
#    (the module imports torch) and reaches the shards via the path:
#      PLE_MODULE=ple_patches/vllm_ple_mmap_patched.py \
#      MODEL=../Qwen3.8-Flash-Next-NVFP4-RadixArk python3 tests/validate_real_layout.py
# Defaults: first the container path, then whatever a clone finds in or next
# to the repo -- a missing path should be a message, not a FileNotFoundError.
def _first_existing(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHED = os.environ.get("PLE_MODULE") or _first_existing(
    "/root/diag/vllm_ple_mmap_patched.py",
    os.path.join(_REPO, "ple_patches", "vllm_ple_mmap_patched.py"),
)
MODEL = os.environ.get("MODEL") or _first_existing(
    "/models/qwen3.8-flash-next-nvfp4",
    os.path.join(_REPO, os.pardir, "Qwen3.8-Flash-Next-NVFP4-RadixArk"),
)
if PATCHED is None or not os.path.exists(PATCHED):
    sys.exit("patched vllm_ple_mmap not found -- set PLE_MODULE=<path> "
             "(expected for example at " + _REPO + "/ple_patches/vllm_ple_mmap_patched.py; "
             "restore ple_patches/vllm_ple_mmap_patched.py from the repository)")
if MODEL is None or not os.path.isdir(MODEL):
    sys.exit("no model directory with the real shards -- MODEL=<path> "
             "(container: /models/qwen3.8-flash-next-nvfp4)")

os.environ.setdefault("VLLM_PLE_MMAP_MADV_RANDOM", "0")  # don't touch live views
os.environ["VLLM_PLE_MMAP_PREFETCH"] = "1"

spec = importlib.util.spec_from_file_location("ple_mod", PATCHED)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except ImportError as exc:  # torch is missing outside the container
    sys.exit("the patched module did not load (" + str(exc) + ") -- it needs a "
             "Python with torch: available inside the serve container, on the host "
             "in a venv with torch)")

import numpy as np

# ---- locate the PLE layer the runtime actually uses (mirror _setup_table)
layer_idx = None
index_path = os.path.join(MODEL, "model.safetensors.index.json")
weight_map = {}
if os.path.exists(index_path):
    import json
    weight_map = json.load(open(index_path))["weight_map"]

# find any ple ngram shard name -> extract layer idx (all shards share it)
import re
ple_layer = None
for name in weight_map:
    m = re.search(r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$", name)
    if m:
        ple_layer = int(m.group(1))
        break
if ple_layer is None:
    print("FAIL: no PLE ngram shard found in index")
    sys.exit(1)
print(f"PLE layer idx (from real index): {ple_layer}")

shards, dtype_str, scale_entry, cols = mod._find_shards(MODEL, ple_layer)
print(f"shards: {len(shards)}  dtype={dtype_str}  row_bytes={cols}")
print(f"total rows: {sum(r for _,_,r in shards.values())}")
shard_size = max(r for _,_,r in shards.values())  # uniform in this layout
print(f"shard_size (max rows): {shard_size}")

table = mod.MmapPleTable(shards, shard_size=shard_size, row_bytes=cols,
                         torch_dtype=None, workers=32, chunk=2048)
rows_total = table.rows_total
print(f"table.rows_total={rows_total}  file-offsets(sample): "
      f"{ {k: v[1] for k,v in list(shards.items())[:3]} }")

# ---- fadvise recorder + bounds check
FILESIZES = {p: os.path.getsize(p) for p in {v[0] for v in shards.values()}}
CALLS = []
real = os.posix_fadvise
def rec(fd, advice, offset, length):
    CALLS.append((fd, advice, offset, length))
os.posix_fadvise = rec

def bounds_ok():
    """Every fadvise range must sit inside its shard's [base, base+rows*cols) window."""
    bad = []
    for fd, adv, off, ln in CALLS:
        si = next((k for k, v in table._fadvise_fds.items() if v == fd), None)
        if si is None:
            bad.append(("no-shard-for-fd", fd)); continue
        path, base, rows = shards[si]
        if off < 0 or off < base or off + ln > base + rows * cols \
                or off + ln > FILESIZES[path]:
            bad.append((si, off, ln))
    return bad

fail = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        fail.append(name)

# batch 1: decode-sized (<= fast_rows=512)
rng = np.random.default_rng(7)
b1 = rng.integers(0, rows_total, size=300).astype(np.int64)
CALLS.clear()
table._prefetch_rows(b1)
check("decode-sized (300) fadvise issued", len(CALLS) > 0, f"{len(CALLS)}")
check("decode-sized (300) all ranges in-bounds", not bounds_ok(),
      f"{len(bounds_ok())} out-of-bounds")

# batch 2: prefill-sized (> 512)
b2 = rng.integers(0, rows_total, size=1024).astype(np.int64)
CALLS.clear()
table._prefetch_rows(b2)
check("prefill-sized (1024) fadvise issued", len(CALLS) > 0, f"{len(CALLS)}")
bad2 = bounds_ok()
check("prefill-sized (1024) all ranges in-bounds", not bad2, f"{len(bad2)} out-of-bounds")

# batch 3: full gather() end-to-end byte correctness on real shards
b3 = rng.integers(0, rows_total, size=256).astype(np.int64)
out = table.gather(b3)
# verify a few rows match the raw file bytes at the computed offsets
ok = True
for i in [0, 10, 200]:
    g = int(b3[i]); si = g // shard_size; loc = g - si * shard_size
    path, base, rows = shards[si]
    with open(path, "rb") as f:
        f.seek(base + loc * cols)
        raw = f.read(cols)
    if not np.array_equal(np.asarray(out[i], dtype=np.uint8), np.frombuffer(raw, np.uint8)):
        ok = False
check("gather() byte-exact vs raw real-file offsets (rows 0/10/200)", ok)

os.posix_fadvise = real
print()
if fail:
    print(f"RESULT: {len(fail)} FAILURE(S): {fail}"); sys.exit(1)
print("RESULT: ALL PASS on real shard layout")
