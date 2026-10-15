# Prefill tuning

Two knobs in vLLM control how prompt tokens are processed:
`--max-num-batched-tokens` (shared step budget) and
`--long-prefill-token-threshold` (per-request cap).

## `--max-num-batched-tokens` (shared step budget)

Env: `QFN_MAX_NUM_BATCHED_TOKENS`; Mia/hybrid override it with
`QFN_MIA_MAX_NUM_BATCHED_TOKENS` (also 8192). Set to **8192** in this deployment; the
vLLM preview default is 8192.

The maximum total number of tokens (prefill AND decode) that the scheduler may
process in a single engine step. This is the **shared step budget for ALL
concurrent requests**: in one step, the tokens of every active request —
new prefill chunks plus ongoing decode tokens — count against the same number.

Trade-off:

- Higher value: a single large prefill finishes in fewer steps (faster TTFT),
  but the concurrent decode streams wait longer for their token in that step
  (larger ITL spike while the prefill runs).
- Lower values may shorten individual blocking steps, but in our tests they
  prolonged prefill and mutual throttling. They did not produce a better
  overall result for our agent workload.

Ignoring decode slots and scheduling overhead, 8192 allows a 64k prompt
to prefill in approximately eight steps. SGLang has separate tuning settings.

To change it, set `QFN_MAX_NUM_BATCHED_TOKENS` for k3 or
`QFN_MIA_MAX_NUM_BATCHED_TOKENS` for Mia/hybrid in `.env` (or as a shell
override) and restart.

## `--long-prefill-token-threshold` (per-request cap)

Env: not set. vLLM default is **0 = disabled**.

A per-request cap: prompts longer than this many tokens are considered "long"
and their prefill chunk is capped at this value per step, regardless of the
shared step budget. With the default 0 the cap is disabled — the shared step
budget alone governs chunk size.

History: this flag was previously set to 1024 (per-request cap) and removed in
an earlier configuration. The combination `MBT=1024` + threshold 1024 formed a
starvation pattern: one long prefill could eat the whole step budget while the
threshold kept it "long", and the other streams starved (the 4-stream collapse;
ABLATION log). The old partial-prefill flags
`--max-num-partial-prefills` / `--max-long-partial-prefills` are missing from
this preview build's CLI (unknown flag → immediate exit 2), so the shared step
budget is the only prefill shaping available.

## Current setting

Shared step budget **8192**, **no per-request cap**.

In our production tests, increasing the budget to 8192 was the change that
clearly improved performance. Prefill effectively takes priority and completes
quickly, at the cost of a short decode trough of approximately **1 token/s**.
The trough lasts from a few seconds to just under a minute, depending on
context length and uncached prompt work. With good prefix caching, agent
loops typically experience only **a few single-digit seconds** of interruption.

This describes the observed effect of the shared token budget, not a separate
scheduler priority flag. The PLE prefetch experiment did not improve this
workload. See [the investigation and outcome](decode-collapse-fix.md) and
[production experience](production-experience.md).

## Overrides

```bash
QFN_MAX_NUM_BATCHED_TOKENS=8192 ./scripts/qwen38f-vllm.sh   # production baseline
QFN_MIA_MAX_NUM_BATCHED_TOKENS=8192 ./scripts/qwen38f-vllm.sh hybrid  # production profile
```
