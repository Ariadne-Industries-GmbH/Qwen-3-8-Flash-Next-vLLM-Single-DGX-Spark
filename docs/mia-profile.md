# mia profile — mechanics

Profile `mia` = `k3` + reduced-vocabulary MTP drafting. The technique comes
from the AGPL-3.0-or-later repo MiaAI-Lab `Qwen3.8-Flash-Next-Single-DGX-Spark`,
adapted to the RadixArk deployment (deliberately not a complete MiaAI image:
their PLE/modelopt/QSA patches fit their Mia checkpoint, not RadixArk — the
blazux image solves that better). Never parallel to k3 (same GPU, same port
44415; switch via `down`).

## Delta vs. `k3`

| Delta vs. `k3` | Effect |
|---|---|
| Reduced MTP draft vocab **65,536** IDs (instead of 248,077) | The MTP drafter carries its own BF16 `lm_head` (1.27 GB) and reads it three times per engine step with MTP3. The patch (`mia/patch_mtp_draft_vocab.py`) slices the head down to the frequency-ranked IDs from `mia/draft_vocab_public.txt`; the argmax runs over the slice (~0.34 GB) → ~2.8 GB less memory bandwidth per step. Tokens outside the slice cannot be proposed by the drafter; the target model verifies each proposed draft using its full vocabulary. The target head remains unchanged. MiaAI measurement (same Spark type): **+26%** (85.9 → 108.1 tok/s) |
| Explicit `cudagraph_capture_sizes [4,8,12,16]` | = (1+K)·S for S=1..4, K=3: every verify batch the scheduler can build has a graph (the vLLM default list lacked 12 → a 3-stream verify would have run eager). PIECEWISE stays mandatory (the PLE-mmap lookup must remain outside the graphs) |
| `VLLM_USE_V2_MODEL_RUNNER=1` | Pins the V2 runner so that the MTP draft config copy does not push it back to V1 |
| Image | `qwen38-flash-mia:2` (mia layer over the k3 image) |

All other parameters (KV 11.60 GiB hard, MTP3, FP8 KV, PLE-mmap, 4x180k,
shared prefill step budget 8192) are 1:1 `k3`.

## Public draft vocabulary

The default is `mia/draft_vocab_public.txt`: 65,536 IDs ranked from the pinned
public English, German and code corpora, with weights 1:3:3. Its provenance
record is included. No model-output capture or private corpus is needed.
See [draft-vocab.md](draft-vocab.md) for inputs and verification.

## A/B test protocol (on the host; only one server may occupy the GPU at a time)

```bash
# k3 baseline
sudo ./scripts/qwen38f-vllm.sh dropcaches
./scripts/qwen38f-vllm.sh k3
./scripts/qwen38f-vllm.sh kv                 # >= 720k KV tokens, >= 4.00x
python3 mia/bench_mixed.py --tag k3          # -> logs/mia-bench.jsonl

# Mia
./scripts/qwen38f-vllm.sh down
sudo ./scripts/qwen38f-vllm.sh dropcaches
./scripts/qwen38f-vllm.sh build-image-mia    # once (~2-5 min)
./scripts/qwen38f-vllm.sh vocab-public       # optional rebuild; public vocabulary is included
./scripts/qwen38f-vllm.sh mia
./scripts/qwen38f-vllm.sh kv mia             # expected: "MTP draft vocab: 65536 of 248077 tokens"
python3 mia/bench_mixed.py --tag mia
```

`mia/bench_mixed.py` (adapted from MiaAI `bench/mixed.py`): two decode streams
+ an injected ~64k prefill, reporting ITL p95/p99 + decode throughput during
the prefill window. Target values from the MiaAI measurement: **p95 < ~700
ms**, decode during prefill **>= 3.45 tok/s** aggregated. Control figure:
`mean_acceptance_length` from the spec metrics — clearly below ~2.7-2.8 would
point to corpus/weights problems.

## License

The baked-in patch and the vocab builder come from MiaAI-Lab
(AGPL-3.0-or-later). Private use is straightforwardly covered; passing on
image/repo/files is subject to AGPL (see the License section of the README).
Details, build mechanics and all files: `mia/README.md`.
