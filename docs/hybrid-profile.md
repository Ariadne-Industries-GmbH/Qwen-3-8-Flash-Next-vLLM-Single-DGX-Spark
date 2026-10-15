# hybrid profile — mechanics

Profile `hybrid` = `mia` + the dense side layers of the existing checkpoint
converted locally to block-wise FP8. The only lever that brings space
**and** speed at the same time.

## FP8 conversion

Measured against the local index (not just copied from the reference):

| Quantity | Value |
|---|---:|
| affected tensors | **300** (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,k,v,o}_proj`, `mlp.shared_expert.{gate,up,down}_proj`) |
| affected shards | **4** (`model-bf16-00001/00010/00011/00012`) |
| BF16 content of these shards | **14.91 GiB** (largest shard 9.99 GiB) |
| after (fp8 + fp32 scales) | ~7.5 GiB → **~7.4 GiB fewer resident weights** |
| NVFP4 routed experts | **untouched** (stay as published) |
| MTP weights / tokenizer | **untouched** → the 65,536-entry draft vocab still applies |

blazux reference (@656cdb0): ~+20% decode, ~8% more KV space, agentic
tournament 45/51 vs. 45/51 against unchanged NVFP4. A controlled A/B measurement for this deployment is still pending.
Our production observations are recorded in
[the production experience report](production-experience.md).

## Procedure (in this order — the server must be down for the conversion;
the converter holds a complete 9.99 GiB shard in RAM)

```bash
./scripts/qwen38f-vllm.sh down
./scripts/qwen38f-vllm.sh prepare-hybrid      # once, ~10 min, CPU-only inside the image
./scripts/qwen38f-vllm.sh build-image-mia     # Mia layer (draft vocab + MADV_RANDOM)
sudo ./scripts/qwen38f-vllm.sh dropcaches
./scripts/qwen38f-vllm.sh hybrid              # port 44415, otherwise identical to mia
./scripts/qwen38f-vllm.sh kv hybrid           # compare KV tokens + resident weights
```

## Layout (hardlink clone, not symlink)

`blazux/prepare-hybrid.sh` clones the HF snapshot via relative symlinks onto
`../../blobs/<sha>` — that only works inside the HF cache hierarchy. Our
checkpoint is real files in a flat folder, so
`Qwen3.8-Flash-Next-NVFP4-RadixArk-fp8hybrid` is created as a **hardlink
clone** (`mia/prepare_hybrid_snapshot.sh`): instant, 0 bytes of extra space,
only ~7.7 GiB of genuinely new files. The hybrid container therefore only has
to mount the snapshot directory (no symlink resolution, no parent mount), and
the snapshot even survives moving the original.

**The only hazard with this layout — mitigated:** `fp8_convert.py` writes
`model.safetensors.index.json` **in place**. As long as the entry were a
hardlink to the original index, that would be the same inode — and the
conversion would bend the original checkpoint's index over to fp8 (a broken
k3/mia setup). The driver therefore moves the index onto its own inode before
the first write and **proves** afterwards that no `weight_scale_inv` tensor
landed in the original and that all four source shards are still there. That
is also why there is no `chmod` on `*.safetensors` (with hardlinks the
original permissions would be affected), and the container runs with
`--user` set to the host user (so the snapshot can be deleted again without
sudo).

## Requirements

- **NEVER** serve the hybrid snapshot without `VLLM_FP8_HYBRID=1` — depending
  on the loader that either gives a load error or silently mishandles the
  mixed NVFP4/FP8 configuration. The profile sets `VLLM_FP8_HYBRID=1` and
  `VLLM_USE_DEEP_GEMM=0` together (same as `blazux/serve.sh`).
- **Not set: `VLLM_FP8_PAD_M4`.** The reference measures roughly -40% TTFT at
  8k with it, **but only with prefix caching disabled**; with
  `--enable-prefix-caching` (the default) the prefill chunks are 1600-aligned
  and the padding is a no-op.

## Where the ~7 GiB go

They flow back into the same unified pool — which primarily grows the **PLE
page cache** (less NVMe refetching; given the current memory situation that is
the bottleneck). The hard KV budget does **not** grow automatically. Anyone
who wants them in KV instead: `QFN_KV_CACHE_MEMORY_BYTES=14495514624`
(13.5 GiB) — but only after a verified hybrid boot.

## Rollback and cleanup

- Rollback: stop with `./scripts/qwen38f-vllm.sh down`, then start
  `./scripts/qwen38f-vllm.sh k3`; k3/mia serve the original checkpoint.
- Clean up the snapshot: `rm -rf ../Qwen3.8-Flash-Next-NVFP4-RadixArk-fp8hybrid`.
