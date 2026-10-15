#!/usr/bin/env python3
"""Unit test for the WILLNEED prefetch patch (Fix A) against SYNTHETIC shards.

Validates the exact risk named in the plan:
  1. The fadvise byte-range math (shard file offset + local_row * row_bytes)
     maps to EXACTLY the bytes of the touched rows, no drift.
  2. Contiguous-run splitting (gap <= 64) works.
  3. gather() results are byte-identical with prefetch ON (no behavior change).
  4. VLLM_PLE_MMAP_PREFETCH=0 is a strict no-op (zero fadvise calls).

Torch is stubbed (only import-time names are needed by vllm_ple_mmap).
"""
import importlib.util
import os
import struct
import sys
import types

# The patched module is a versioned runtime dependency of the hybrid profile.
# PLE_MODULE can point at an alternative module for validation.
HERE = os.path.dirname(os.path.abspath(__file__))
PATCHED = os.environ.get(
    "PLE_MODULE",
    os.path.join(HERE, "..", "ple_patches", "vllm_ple_mmap_patched.py"),
)
if not os.path.isfile(PATCHED):
    sys.exit(f"patched module not found: {PATCHED}\n"
             f"  restore ple_patches/vllm_ple_mmap_patched.py or set PLE_MODULE=/path/module.py")

# ---------------------------------------------------------------- stub torch
torch_stub = types.ModuleType("torch")
torch_stub.float8_e4m3fn = "f8e4m3"
torch_stub.float8_e5m2 = "f8e5m2"
torch_stub.bfloat16 = "bf16"
torch_stub.float16 = "f16"
torch_stub.empty = lambda *a, **k: None
nn_stub = types.ModuleType("torch.nn")


class _Module:
    def __init__(self, *a, **k):
        pass


nn_stub.Module = _Module
torch_stub.nn = nn_stub
sys.modules["torch"] = torch_stub
sys.modules["torch.nn"] = nn_stub

os.environ["VLLM_PLE_MMAP_PREFETCH"] = "1"
os.environ["VLLM_PLE_MMAP_MADV_RANDOM"] = "0"  # keep test files pristine

spec = importlib.util.spec_from_file_location("ple_mod", PATCHED)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ------------------------------------------------------- synthetic shard files
ROW_BYTES = 8
SHARD_ROWS = 200          # > 64 gap so we can test run-splitting
HEADER = 16 * b"\xAA"     # fake header -> offset != 0 (range-guard case)


def row_payload(s, r):
    """Distinct, deterministic payload per (shard, local_row)."""
    v = s * 1000 + r
    return bytes([v % 251, (v * 7) % 251, (v * 13) % 251, (v * 31) % 251,
                  (v * 3) % 251, (v * 5) % 251, (v * 17) % 251, (v * 11) % 251])


def make_shard(path, shard_id):
    with open(path, "wb") as f:
        f.write(HEADER)
        for r in range(SHARD_ROWS):
            f.write(row_payload(shard_id, r))


import tempfile
tmp = tempfile.mkdtemp(prefix="ple_shard_test_")
p0 = os.path.join(tmp, "shard0.bin")
p1 = os.path.join(tmp, "shard1.bin")
make_shard(p0, 0)
make_shard(p1, 1)

# fadvise recorder
CALLS = []
real_fadvise = os.posix_fadvise


def fake_fadvise(fd, advice, offset, length):
    CALLS.append((fd, advice, offset, length))


os.posix_fadvise = fake_fadvise

shards = {0: (p0, len(HEADER), SHARD_ROWS), 1: (p1, len(HEADER), SHARD_ROWS)}
table = mod.MmapPleTable(shards, shard_size=SHARD_ROWS, row_bytes=ROW_BYTES,
                         torch_dtype=None, workers=4, chunk=64)

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------- test 1:
# gather correctness with prefetch ON (inline fast path, ids <= fast_rows)
import numpy as np
ids = np.array([3, 0, 5, 11, 12, 13, 19], dtype=np.int64)
out = table.gather(ids)
# global row id -> shard = id // shard_size ; local = id % shard_size
expected = np.array([list(row_payload(int(i // SHARD_ROWS), int(i % SHARD_ROWS))) for i in ids],
                    dtype=np.uint8)
check("gather byte-exact (fast path, prefetch ON)", out.shape == (7, ROW_BYTES)
      and np.array_equal(out, expected))

# verify recorded fadvise ranges map EXACTLY onto the touched row bytes
def verify_ranges():
    # test-1 ids < 200 -> shard 0 only; local rows {0,3,5,11,12,13,19},
    # all gaps <= 64 -> exactly one run 0..19; shard 1 stays untouched.
    expect_s0 = [(len(HEADER) + 0 * ROW_BYTES, (19 - 0 + 1) * ROW_BYTES)]
    expect_s1 = []
    data0 = open(p0, "rb").read()
    data1 = open(p1, "rb").read()

    def rowspan(data, off, ln, sid_):
        """All rows fully covered, in order, with no partial-row drift."""
        if (off - len(HEADER)) % ROW_BYTES or ln % ROW_BYTES:
            return False
        got = data[off:off + ln]
        r0 = (off - len(HEADER)) // ROW_BYTES
        ok = True
        for k in range(ln // ROW_BYTES):
            if got[k * ROW_BYTES:(k + 1) * ROW_BYTES] != row_payload(sid_, r0 + k):
                ok = False
        return ok

    by_fd = {}
    for fd, adv, off, ln in CALLS:
        by_fd.setdefault(fd, []).append((off, ln))
    # fds: table._fadvise_fds[0] / [1]
    for sid, expect in ((0, expect_s0), (1, expect_s1)):
        fd = table._fadvise_fds.get(sid)
        got = sorted(by_fd.get(fd, []))
        if got != sorted(expect):
            return False, f"shard{sid}: got {got} expected {sorted(expect)}"
        for off, ln in expect:
            if not rowspan(data0 if sid == 0 else data1, off, ln, sid):
                return False, f"shard{sid}: range ({off},{ln}) covers wrong bytes"
    return True, ""

ok, detail = verify_ranges()
check("fadvise ranges == exact touched-row bytes (shard0+shard1)", ok, detail)

# ------------------------------------------------------------------- test 2:
# run-splitting across a gap > 64 rows (shard 0: rows 7 and 100)
CALLS.clear()
ids2 = np.array([7, 100, 8], dtype=np.int64)  # local 7,8 (one run) + 100 (own run)
table._prefetch_rows(ids2)
fd0 = table._fadvise_fds[0]
got0 = sorted(c for c in CALLS if c[0] == fd0)
exp0 = sorted([(fd0, 3, len(HEADER) + 7 * ROW_BYTES, 2 * ROW_BYTES),
               (fd0, 3, len(HEADER) + 100 * ROW_BYTES, 1 * ROW_BYTES)])
check("gap>64 splits into separate runs", got0 == exp0, f"got {got0} exp {exp0}")

# ------------------------------------------------------------------- test 3:
# prefetch OFF => zero fadvise calls (behavioral identity to old code path)
CALLS.clear()
os.environ["VLLM_PLE_MMAP_PREFETCH"] = "0"
out_off = table.gather(np.array([4, 20, 21, 199], dtype=np.int64))
os.environ["VLLM_PLE_MMAP_PREFETCH"] = "1"
check("PREFETCH=0 is a strict no-op (0 fadvise calls)", len(CALLS) == 0,
      f"{len(CALLS)} calls")
check("gather identical with PREFETCH=0",
      np.array_equal(out_off, np.array([list(row_payload(int(i // SHARD_ROWS), int(i % SHARD_ROWS))) for i in [4, 20, 21, 199]], dtype=np.uint8)))

# ------------------------------------------------------------------- test 4:
# pooled path (ids > fast_rows=512) also prefetches and stays byte-exact
CALLS.clear()
big = np.random.default_rng(42).integers(0, 2 * SHARD_ROWS, size=700).astype(np.int64)
out_big = table.gather(big)
exp_big = np.array([list(row_payload(int(i // SHARD_ROWS), int(i % SHARD_ROWS))) for i in big],
                   dtype=np.uint8)
check("pooled path (700 ids) byte-exact", np.array_equal(out_big, exp_big))
check("pooled path issued fadvise", len(CALLS) > 0, "no calls recorded")
# every fadvise range must align to ROW_BYTES and stay within its shard file
size = len(HEADER) + SHARD_ROWS * ROW_BYTES
bad = [c for c in CALLS if c[2] < len(HEADER) or c[2] + c[3] > size
       or (c[2] - len(HEADER)) % ROW_BYTES or c[3] % ROW_BYTES]
check("all fadvise ranges row-aligned & in-bounds", not bad, f"{len(bad)} bad")

os.posix_fadvise = real_fadvise
print()
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("RESULT: ALL PASS")
