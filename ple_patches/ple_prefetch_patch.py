#!/usr/bin/env python3
"""Backport: Mia's posix_fadvise(WILLNEED) batched prefetch for the PLE mmap table.

Deployment result: tested without an observed production performance gain.
The effective change for prefill/decode interference was the 8192-token shared
scheduler budget, not this patch. See docs/decode-collapse-fix.md.

Target: /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py.
The hybrid profile bind-mounts the versioned patched module so it survives
container recreation. The upstream cold-gather microbenchmark is not evidence
of a production speedup in this deployment.

What it does:
  * In MmapPleTable._gather: BEFORE the (inline or pooled) row copy, issue
    posix_fadvise(fd, FADV_WILLNEED) for each shard's touched byte ranges
    (contiguous runs of the sorted unique row indices). The kernel issues the
    reads ahead of the numpy copy; this was intended to reduce serial
    page-fault latency but did not improve our production workload.
  * Covers BOTH paths: the inline fast path (ids.size <= fast_rows, the
    DECODE-sized path that runs synchronously on the EngineCore main thread)
    and the pooled path (large prefill batches).
  * Guarded by VLLM_PLE_MMAP_PREFETCH (default 0 = OFF; "1" enables at runtime
    and keeps fds open. Enable via env when (re)starting the container, or set
    to 1 in .env for the serve script.)

Apply (planned restart window; no-op if already patched):

  DOCKER_HOST=unix:///run/docker.sock docker cp ple_prefetch_patch.py q38f-hybrid-vllm:/root/diag/
  DOCKER_HOST=unix:///run/docker.sock docker exec q38f-hybrid-vllm python3 /root/diag/ple_prefetch_patch.py

Validation: the tests in tests/ check byte ranges and unchanged gather output.
They do not demonstrate a throughput improvement.
"""
import re
import shutil
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py"

# --- anchor 1: fast (inline, decode-sized) path -----------------------------
ANCHOR_FAST = """        if ids.size <= self.fast_rows:
            # Decode-sized batches: thread-pool dispatch costs more than the
            # reads themselves (~50 tasks for ~65 rows). Gather inline instead.
            if ids.min() < 0 or ids.max() >= self.rows_total:
                raise IndexError(
                    f"PLE row id out of range: [{ids.min()}, {ids.max()}] "
                    f"for {self.rows_total} rows"
                )
            shard = ids // self.shard_size
            local = ids - shard * self.shard_size
            out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)"""

PATCHED_FAST = """        if ids.size <= self.fast_rows:
            # Decode-sized batches: thread-pool dispatch costs more than the
            # reads themselves (~50 tasks for ~65 rows). Gather inline instead.
            if ids.min() < 0 or ids.max() >= self.rows_total:
                raise IndexError(
                    f"PLE row id out of range: [{ids.min()}, {ids.max()}] "
                    f"for {self.rows_total} rows"
                )
            self._prefetch_rows(ids)
            shard = ids // self.shard_size
            local = ids - shard * self.shard_size
            out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)"""

# --- anchor 2: pooled path (after dedup, before task dispatch) --------------
ANCHOR_POOL = """        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows_total:
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size"""

PATCHED_POOL = """        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows_total:
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        self._prefetch_rows(uniq)
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size"""

# --- new method: inserted before prewarm() ----------------------------------
PREFETCH_METHOD = '''    def _prefetch_rows(self, ids) -> None:
        """Mia's fix: posix_fadvise(WILLNEED) over the sorted touched rows.

        Contiguous runs of local row indices per shard (gap <= 64 rows kept
        together) are mapped to byte ranges and fadvise'd; the kernel issues
        the reads ahead of the numpy copy. Upstream measurement: cold 280-row
        gather ~20 ms -> ~1.51 ms. No-op when VLLM_PLE_MMAP_PREFETCH=0 or no
        fds are available.
        """
        fds = getattr(self, "_fadvise_fds", None)
        if not fds or _env_int("VLLM_PLE_MMAP_PREFETCH", 0) <= 0:
            return
        import os as _os

        WILLNEED = getattr(_os, "FADV_WILLNEED", 3)
        srt = np.sort(ids)
        shard = srt // self.shard_size
        local = srt - shard * self.shard_size
        for si in np.unique(shard):
            fd = fds.get(int(si))
            if fd is None:
                continue
            sel = local[shard == si]
            breaks = np.flatnonzero(np.diff(sel) > 64)
            starts = np.concatenate(([sel[0]], sel[breaks + 1]))
            ends = np.concatenate((sel[breaks], [sel[-1]]))
            base = self._fadvise_offsets[int(si)]
            for a, b in zip(starts.tolist(), ends.tolist()):
                off = base + int(a) * self.row_bytes
                length = (int(b) - int(a) + 1) * self.row_bytes
                try:
                    _os.posix_fadvise(fd, WILLNEED, off, length)
                except (OSError, AttributeError):
                    break

    '''

# --- __init__ extension ------------------------------------------------------
ANCHOR_INIT = """        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)"""

PATCHED_INIT = """        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)
        # WILLNEED prefetch (Mia backport): keep raw fds + shard file offsets
        # for posix_fadvise. O_RDONLY does not disturb the memmap views.
        self._fadvise_fds: dict[int, int] = {}
        self._fadvise_offsets: dict[int, int] = {}
        import os as _os

        for si, (path, off, _rows) in shards.items():
            try:
                self._fadvise_fds[si] = _os.open(path, _os.O_RDONLY)
                self._fadvise_offsets[si] = int(off)
            except OSError:
                pass"""


def apply_to(src: str) -> str:
    if "WILLNEED" in src:
        return src  # already patched
    for anchor in (ANCHOR_FAST, ANCHOR_POOL, ANCHOR_INIT):
        if anchor not in src:
            raise SystemExit("ERROR: anchor not found:\n" + anchor[:120] + "...")
    src = src.replace(ANCHOR_FAST, PATCHED_FAST, 1)
    src = src.replace(ANCHOR_POOL, PATCHED_POOL, 1)
    src = src.replace(ANCHOR_INIT, PATCHED_INIT, 1)
    if "    def prewarm(self) -> None:" not in src:
        raise SystemExit("ERROR: prewarm() anchor not found")
    src = src.replace("    def prewarm(self) -> None:",
                      PREFETCH_METHOD.rstrip() + "\n    def prewarm(self) -> None:", 1)
    return src


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PATH
    src = open(path).read()
    out = apply_to(src)
    if out is src:
        print("already patched, nothing to do")
        return
    shutil.copyfile(path, path + ".bak")
    open(path, "w").write(out)
    print(f"patched OK (backup at {path}.bak)")


if __name__ == "__main__":
    main()
