# Ariadne Engine production experience

We run the
[Ariadne Engine](https://github.com/Ariadne-Industries-GmbH/Ariadne-Engine) in production on a single NVIDIA DGX Spark.
Our agents run stably for hours and complete real tasks. Context windows in our
workload range from approximately 100,000 to 160,000 tokens. These figures come from our
day-to-day production use rather than a standardized benchmark. The figures
do not assert that every stream simultaneously held a fully occupied context
window of that size.

| Serving setup | Single stream | Two streams, aggregate | Four streams, aggregate |
|---|---:|---:|---:|
| vLLM hybrid, reduced draft vocabulary | ~25–40 tokens/s | up to ~60 tokens/s | ~70–85 tokens/s |
| SGLang, tested configuration | ~20–25 tokens/s | ~30–40 tokens/s | up to ~50-60 tokens/s |

The vLLM observations apply to the hybrid profile with the documented settings,
reduced draft vocabulary and an 8192-token shared scheduler budget.
The SGLang observations apply to the previously tested active configuration:
4096-token prefill chunks and a 16384-token total prefill budget. No changed
SGLang tuning is implied by this report.

Prefill is acceptably fast for this deployment's tasks. Optimized prefix
caching is crucial for fast agent loops at long contexts: preserving reusable
prompt prefixes avoids repeatedly prefilling most of a long conversation.
A cold prompt and a cache hit can therefore have very different latency.

## Prefill and temporary decode slowdowns

In our vLLM tests, raising the shared scheduler budget to **8192 tokens** was
the only change in the decode-collapse investigation that clearly improved
performance. Prefill effectively takes priority and finishes quickly, while
decode temporarily drops to approximately **1 token/s**. Depending on context
length and uncached prompt work, this lasts from a few seconds to just under
a minute. With good prefix caching, our agent loops usually see only **a few
single-digit seconds** of interruption.

In our tested SGLang setup, prefill and decode run in separate phases and
batches, without the same mutual throttling. Decode waits for prefill, so the
temporal effect is comparable to our current vLLM production setup. This
comparison does not imply a change to SGLang's tested configuration.

The PLE prefetch experiment did not produce a performance improvement in our
vLLM workload. The higher token budget, together with effective prefix caching,
accounts for the improvement we observed.

Aggregate throughput is the total across concurrent streams, not the speed
of each request. The reported ranges depend on the task, context and cache
state; they are not minimum performance commitments.
