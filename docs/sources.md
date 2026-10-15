# Sources (verified 2026-09-07)

- **blazux/qwen3.8-Flash-DGX** — README, Dockerfile, `scripts/serve.sh`
  (pin `656cdb075b57671c9bd6807b57f630fb3c1641fe`): RadixArk checkpoint as
  reference model, PLE-mmap, FP8-KV patch, det-Top-k, flags.
- **MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark** — README, `.env.sample`:
  KV measurements (FP8 ~1.13M tokens @ ~16 GiB), K-sweep (K=3 optimal), MTP3,
  BF16 S-state (+8.5% decode), prefill table, co-hosting warnings.
- **RadixArk/Qwen3.8-Flash-Next-NVFP4** (Hugging Face) — local checkpoint:
  `config.json` (modelopt NVFP4, 31 `mtp.*` tensors, 48 GB PLE),
  `chat_template.jinja` (reasoning_effort: only xhigh/medium/low).
- **vLLM (main)** `vllm/config/cache.py` — `kv_cache_memory_bytes` ignores
  `gpu_memory_utilization` (KV sizing); the startup pool check stays
  effective.
