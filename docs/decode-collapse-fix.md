# Decode slowdowns during prefill: what improved production performance

Increasing the shared scheduler token budget to **8192** was the only change
in our decode-collapse investigation that produced a clear performance
improvement. The PLE prefetch patch and diagnostic changes described below
were tested, but did not improve this workload. We do not regard them as
relevant performance fixes for the behaviour described here.

## What changed with 8192 batched tokens

With the smaller token budgets we tested, prefill and decode throttled each
other and prolonged the slowdown. At **8192**, prefill effectively takes
priority in our production workload: the engine finishes it quickly, then
returns to normal decode throughput.

This still causes a short decode trough of approximately **1 token/s** while
prefill runs. Depending on context length and how much of the prompt must be
prefilled, this lasts from a few seconds to just under a minute. Our production
agent workloads use context windows of **100k–160k tokens**; see
[production experience](production-experience.md).

With effective prefix caching, most of a long agent conversation is reused.
Only the newly appended prompt suffix needs prefill, so the decode trough in
these agent loops usually lasts just **a few single-digit seconds**. This
short interruption is the trade-off that worked well for us: finish prefill
and get back to decode, instead of stretching both phases out.

The configuration uses:

```dotenv
QFN_MAX_NUM_BATCHED_TOKENS=8192
QFN_MIA_MAX_NUM_BATCHED_TOKENS=8192
```

These variables set the shared budget for prefill and decode tokens per
scheduler step. They do not configure a separate prefill-priority scheduling
policy; the effective prioritization above describes what we observed with
this workload. See [prefill tuning](prefill-tuning.md).

## Comparison with SGLang

In our tests of the documented SGLang setup, prefill and decode run in separate
phases and batches. We did not observe the same mutual throttling there.
Decode still waits while prefill runs, so the timing experienced by an agent
is comparable to our current vLLM production setup with an 8192-token budget.
This observation applies to our tested configuration, not every SGLang mode.

Both setups therefore benefit strongly from prefix caching: reusing a long
prefix reduces the new prefill work and shortens the interruption to agent
loops. SGLang retains its tested 4096-token chunk setting; this comparison
calls for no change to its configuration.

## Historical PLE diagnosis: not confirmed as the solution

The earlier investigation suspected serial page-fault latency in the PLE mmap
gather path. That motivated the batched `FADV_WILLNEED` prefetch experiment.
Testing it did **not** establish a production performance benefit or resolve
the observed slowdown. The previous predictions of lower page-fault rates,
1–5 ms gather times and 20–27% faster decode must not be read as achieved
results for this deployment.

## Tested fixes: no demonstrated performance benefit

- **Batched PLE prefetch (formerly Fix A):**
  `ple_patches/vllm_ple_mmap_patched.py` uses
  `posix_fadvise(FADV_WILLNEED)` for touched row ranges. The patch was tested
  without a performance improvement in our workload. The upstream cold-gather
  microbenchmark did not translate into an observed production gain here.
  The current hybrid configuration still includes the module and sets
  `QFN_PLE_MMAP_PREFETCH=1`; its presence is not the explanation for the
  production improvement. `QFN_PLE_MMAP_PREFETCH=0` disables the prefetch path.
- **Iteration-detail logging (formerly Fix C):**
  `--enable-logging-iteration-details` provided diagnostic timings. It did not
  improve performance and is not included in the current launch flags.
- **MTP ablation (formerly proposed Fix B):** this was a proposed follow-up,
  not an established fix. The production profile continues to use MTP3.

There is no demonstrated need to deploy these experiments to address the
prefill/decode slowdown. The effective change was the higher scheduler token
budget, combined with prefix caching for responsive agent loops.

## What the patch tests actually validate

`tests/test_prefetch_math.py` checks synthetic shard offsets, range splitting,
unchanged gather output and the disabled prefetch path.
`tests/validate_real_layout.py` checks range bounds and gather output against
the real checkpoint layout. These checks establish correctness of the patch's
file access, not a throughput improvement.

The previously recorded results were 8/8 synthetic checks and 5/5 real-layout
checks. They are historical correctness results, not fresh performance
measurements. The real-layout check needs the checkpoint, `torch` and `numpy`:

```bash
PLE_MODULE=ple_patches/vllm_ple_mmap_patched.py \
MODEL=../Qwen3.8-Flash-Next-NVFP4-RadixArk python3 tests/validate_real_layout.py
```
