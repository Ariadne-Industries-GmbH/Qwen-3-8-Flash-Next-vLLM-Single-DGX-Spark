# Qwen3.8-Flash-Next on DGX Spark — vLLM

Serve the RadixArk NVFP4 checkpoint on one DGX Spark (GB10, approximately
121.7 GiB unified memory). The production profile is **hybrid**: FP8 side
layers, reduced-vocabulary MTP3 drafting, FP8 KV cache and NVMe-backed PLE.
The OpenAI-compatible endpoint is `http://127.0.0.1:44415/v1`, model ID
`qwen3.8-flash-next`.

Use this deployment for the [Ariadne Engine](https://github.com/Ariadne-Industries-GmbH/Ariadne-Engine) on your
**NVIDIA DGX Spark**. Published by [Ariadne Industries](https://github.com/Ariadne-Industries-GmbH).
After following the setup below, copy the supplied `model_config.json` into
your Ariadne Engine installation to use this deployment. **The configuration
file is pending and will be added before publication.**

Companion deployment: [SGLang on DGX Spark](https://github.com/Ariadne-Industries-GmbH/Qwen-3-8-Flash-Next-SGLang-Single-DGX-Spark).

## Experience in production

We use the documented **hybrid** profile with its current settings and reduced
draft vocabulary to run the Ariadne Engine in production on one NVIDIA DGX
Spark. Our agents run stably for hours and complete real tasks with context
windows of **100k–160k tokens**.

We observe approximately **70–85 tokens/s in aggregate across four concurrent
requests**, **up to 60 tokens/s across two streams**, and **25–40 tokens/s for
a single stream**. Prefill is fast enough for our workload. Optimized prefix
caching is essential to keep agent loops responsive at long contexts.

These figures come from our day-to-day production use rather than a controlled
benchmark. See [our production experience](docs/production-experience.md)
for details.

## Prerequisites

Docker with Compose v2 and NVIDIA GPU support, `git`, `python3`, `curl`,
`sudo` for cache clearing, and the Hugging Face CLI (`hf`). Allow approximately
135 GB for the checkpoint, another 8 GB for hybrid conversion, and space for
container images and build caches. Store the checkpoint on local NVMe.

## Start the production profile

Run from this repository directory:

```bash
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir ../Qwen3.8-Flash-Next-NVFP4-RadixArk
cp .env.example .env
./scripts/qwen38f-vllm.sh build-image
./scripts/qwen38f-vllm.sh build-image-mia
./scripts/qwen38f-vllm.sh prepare-hybrid   # once, with all servers stopped
sudo ./scripts/qwen38f-vllm.sh dropcaches
./scripts/qwen38f-vllm.sh hybrid
# Use a second terminal below, or press Ctrl+C to stop following logs.
./scripts/qwen38f-vllm.sh kv hybrid
./scripts/qwen38f-vllm.sh test
```

The start command follows the detached container logs; Ctrl+C stops the log
view while the server keeps running. You can reattach with
`./scripts/qwen38f-vllm.sh logs hybrid`. First startup can
take 10–15 minutes. Check for at least 720,000 KV tokens, concurrency at least
4.00x for 180,000 tokens, and a successful completion. The public 65,536-ID
draft vocabulary and required PLE patches are included; no corpus build is
needed to serve the model. See [QUICKSTART.md](QUICKSTART.md) for troubleshooting.

## Profiles and configuration

| Profile | Configuration | Use |
|---|---|---|
| `k3` | Original checkpoint, full draft head | Baseline; launcher default when no profile is supplied |
| `mia` | Original checkpoint, public reduced draft vocabulary | Compare vocabulary reduction |
| `hybrid` | Mia drafting plus locally converted FP8 side layers | Production setup |

All profiles share the GPU and port. Stop with `./scripts/qwen38f-vllm.sh down`
before switching profiles or converting the checkpoint. Never clear the page
cache while serving requests.

| Setting | Default |
|---|---|
| Context per request / concurrent requests | `180000` / `4` |
| MTP speculative tokens | `3` |
| KV cache / recurrent state dtype | `fp8_e4m3` / `bfloat16` |
| Hard KV budget | `12459440000` bytes |
| Startup memory utilization | `0.80` |
| Shared scheduler token budget, all profiles | `8192` |
| Per-request long-prefill threshold | unset |
| Draft vocabulary for Mia and hybrid | `mia/draft_vocab_public.txt` |

The two scheduler variables are `QFN_MAX_NUM_BATCHED_TOKENS=8192` and
`QFN_MIA_MAX_NUM_BATCHED_TOKENS=8192`. Shell environment overrides take
precedence over `.env`; see [.env.example](.env.example) for the full template.

## Operations

```bash
./scripts/qwen38f-vllm.sh ps hybrid
./scripts/qwen38f-vllm.sh logs hybrid
./scripts/qwen38f-vllm.sh kv hybrid
./scripts/qwen38f-vllm.sh test
./scripts/qwen38f-vllm.sh down
./scripts/host-tuning.sh status
```

## Further documentation

- [Performance reference](docs/performance-reference.md): measurements and their limits.
- [Memory management](docs/memory-management.md): pool sizing and optional host tuning.
- [Prefill tuning](docs/prefill-tuning.md): scheduler budget and latency trade-offs.
- [Hybrid profile](docs/hybrid-profile.md): conversion, layout and rollback.
- [Mia profile](docs/mia-profile.md): reduced draft head and A/B protocol.
- [Draft vocabulary](docs/draft-vocab.md): public inputs, provenance and verification.
- [Prefill/decode slowdowns](docs/decode-collapse-fix.md): effective token-budget tuning and unsuccessful patch experiments.
- [Reasoning effort](docs/reasoning-effort.md): supported request parameters.
- [Sources](docs/sources.md): upstream references and image pins.

## License

This repository is licensed under [AGPL-3.0-or-later](LICENSE).
Upstream patches retain their attribution notices. Sources and attribution
for the public draft vocabulary are documented in
[docs/draft-vocab.md](docs/draft-vocab.md) and its accompanying provenance file.

<!-- Maintainer: replace the companion repository URL before publication. -->
[companion-deployment]: https://github.com/Ariadne-Industries-GmbH/REPLACE_WITH_SGLANG_REPOSITORY
