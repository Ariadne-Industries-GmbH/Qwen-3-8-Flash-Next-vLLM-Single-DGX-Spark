# Memory management

## Unified pool budget (as of 2026-09-10)

The shortage arises in the **unified pool** (measured 121.69 GiB): Flash Next
and the other host processes all allocate from the same LPDDR5X stock.
"CUDA out of memory" here means "allocation from the pool failed" — regardless
of whether the context is CPU or CUDA.

| Item | GiB |
|---|---:|
| Flash Next steady (78.2 resident weights + 11.60 KV FP8 + ~1.5 MTP3 + ~4-6 CUDA graphs/activations) | ~95-97 |
| Other host processes (OS + co-running services) | ~15 |
| **Sum expected** | **~110-112** |
| Pool | 121.69 |
| **Headroom** | **~10** |

The 2 GiB **display reserved memory** in the BIOS are an extra lever (default
2 GB, selectable between 2 and 4 GB since the July 2026 update; a documented
0 GB mode does not exist). This host is already at 2 GB — do not count them
twice.

### Lever status

| Lever | Gain | Status |
|---|---:|---|
| KV trim 14.1 → 11.60 GiB (4x180k instead of 4x230k) | ~2.5 steady + peak | done |
| **FP8 hybrid for the side layers** | **~7.4 resident**, ~+20% decode | prepared; conversion with the server down (`prepare-hybrid`) |
| PLE mmap `MADV_RANDOM` | up to ~2 page cache | in the mia image; functionally tested on numpy 2.2.6 |
| `--mamba-ssm-cache-dtype bfloat16` | ~0.4-0.6 at 4 streams | done (k3 baseline) |
| Shared prefill/decode step budget | workload-dependent activation peak | production setting 8192 across all profiles |
| BIOS display reserve 4 → 2 GB | 2 | set in the host BIOS (not via the repo) |
| host VM tuning (`scripts/host-tuning.sh`) | **no free GiB** | optional, see below |
| disabling CUDA graphs | ~0.12 | deliberately **not** (decode loss) |

The hybrid lever is the only one that brings space **and** speed at the same
time (see `hybrid-profile.md`); the conversion is manual work with the server
down, because the converter holds a complete 9.99 GiB shard in RAM.

## Host VM tuning is not a free-memory trick

`scripts/host-tuning.sh` sets `vm.min_free_kbytes=4 GiB`,
`vm.watermark_scale_factor=300` (3%) and `vm.swappiness=30` (MiaAI-Lab
`sysctl-spark3.conf`; source/license in the script header). It gains **no** GiB
for vLLM — the reserve is memory that vLLM must no longer touch. It gains
**lead time**: kswapd intervenes at 3% instead of 0.1%, instead of the GPU
driver losing with `NV_ERR_NO_MEMORY` at ~3 GiB MemFree. All three values can
be moved via env override (`sudo MIN_FREE_KBYTES=2097152 ... apply`).

Two consequences to know:

1. `MemAvailable` is then reported ~11 GiB **lower** (watermark formula:
   anything below `high` no longer counts as available). That is the reading,
   not availability — recalibrate watchdog thresholds after the first idle run.
   `revert` restores swappiness/watermark.
2. For that reason `apply` refuses below 8 GiB MemAvailable: the reserve is a
   **before** tool, not a during tool. Order while idle:
   `down` → `sudo ./scripts/host-tuning.sh apply` → `dropcaches` → `up`.

**Do not disable swap.** NVIDIA documents for the Spark that the unified memory
frees DRAM through reclaim/swap; without this safety reserve, "something is
lagging" quickly becomes "CUDA allocation failed". A full swap (e.g. 16/16 GiB)
is not the same as thrashing — the decision goes by the **activity**
(`si`/`so` in `vmstat`, both in the `status` branch). Live sample 2026-09-10
with k3 running: `si`/`so` ~ 0 at `bi` 46-99 MB/s, i.e. old paging plus PLE
refetching from NVMe. That is not a swap problem but a space shortage in the
same pool — and therefore the hybrid lever again.

## KV cache calculation (11.60 GiB — why 11.60 and not 14.1)

The KV pool is committed byte by byte via `--kv-cache-memory-bytes` (KV sizing
ignores `gpu_memory_utilization`; only the startup pool check uses it). On
Flash Next the KV footprint per token is hybrid (12 full-attention layers +
36 GDN layers + indexer ring) and is better evidenced by reference measurements
than by our own calculation:

```text
MiaAI-Lab (same Spark type, V2 runner, MTP3, FP8 KV, BF16 SSM state):
  15.98 GiB pool -> 1,132,586 KV tokens (measured, 2026-09-06, stable over 10x starts)
  ~18.5-19 GiB pool -> 1,216,000-1,222,000 tokens (measured)

from that: ~70k-71k KV tokens/GiB (FP8 KV)

data point (own boot, 2026-09-08, identical image/config):
  16 GiB = 17,179,869,184 bytes -> 1,047,937 tokens = 5.82x at 180k (16,394 B/token)
```

Target: 4 x 180,000 tokens = 720,000 tokens resident. The budget is sized to
4 x 190,000 = 760,000 tokens (190k = accepted upper bound per stream, ~5%
margin over the 180k working context) — every GiB of KV is missing from the
PLE page cache and other host processes on the unified-memory stack. The client caps
requests at 4 anyway. The old 14.1 GiB (4x230k) was ~923.5k tokens and
therefore ~2.5 GiB oversized:

```text
11.60 GiB = 12,459,440,000 bytes -> exactly 760,000 tokens = 4.22x
(760,000 x 16,394 B/token; reserve ~40k tokens)
```

vLLM suffix case (important for manual overrides): `12g` = 12,000,000,000
bytes = 11.18 GiB; `12G` = 12 x 1024^3 = 12.00 GiB. That is why the exact
byte value is set in `.env`.

After startup `./scripts/qwen38f-vllm.sh kv` shows the vLLM line
`GPU KV cache size: ...` (in tokens) and `Maximum concurrency for 180000
tokens per request: X.XXx`.

### Acceptance criteria (hard limit)

- reported KV capacity **>= 720k tokens** (exactly 4 x 180k; budget 4 x 190k,
  reserve ~40k tokens for block alignment/draft)
- `Maximum concurrency for 180000 tokens per request: >= 4.00x`
- `Application startup complete`, no RuntimeError/ValueError
- no `mamba state-copy guard: N out-of-range` in the log (patch regression)

If the KV capacity falls below 720k: `QFN_KV_CACHE_MEMORY_BYTES=14495514624`
(13.5 GiB, ~884k tokens) or `15032385536` (14 GiB, ~917k tokens) — unified
memory allows it, but every GiB is missing from the PLE page cache and
other host processes.

## Memory prerequisites

This deployment plans a large share of the host: the pool check demands
~97.4 GiB (0.80 x 121.7 GiB) free at startup, plus the PLE page cache during
operation. Make sure the host has that headroom before starting. Run
`dropcaches` first so weight leftovers from an earlier run of this server
do not fail the check.

**dropcaches note:** `dropcaches` before startup clears weight leftovers from
the page cache (pool check). Do **not** evict the PLE page cache (48 GB,
accessed via mmap during inference) after startup — prefill speed depends on
the warm cache (reference: first request after a cold cache ~3x slower, 8k row).
