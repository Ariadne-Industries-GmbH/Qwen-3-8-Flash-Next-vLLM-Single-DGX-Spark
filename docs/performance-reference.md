# Performance reference (measured, same Spark type)

Upstream reference measurements, not a guarantee — hardware, image pin and
workload shape matter. For our production observations, see
[the Ariadne Engine experience report](production-experience.md).

## MiaAI-Lab reference (NVFP4, PLE offload, MTP3, FP8 KV, BF16 recurrent state)

600-token prose outputs on short contexts:

| Streams | aggregated | per stream |
|---:|---:|---:|
| 1 | 48.7 tok/s | 48.7 |
| 4 | 113.7 tok/s | 28.4 |
| 8 | 162.9 tok/s | 20.4 |

## Realistic expectation for 4x180k decode

For sessions filled to 4 x 180k that decode at the same time, the realistic
expectation is **~20-25 tok/s per stream** (extrapolation from the reference;
during decode QSA does not read the full 180k KV the way full attention does).

## Decode under concurrent long prefills

In our current production setup, an 8192-token scheduler budget lets prefill
finish quickly, with decode temporarily dropping to approximately **1 token/s**.
The interruption lasts from a few seconds to just under a minute depending on
context length and uncached prompt work. With good prefix caching, our agent
loops usually see only a few single-digit seconds of interruption.

Smaller budgets prolonged mutual throttling in our tests; the PLE prefetch
patch did not improve this behaviour. See [the investigation](decode-collapse-fix.md)
and [our production experience](production-experience.md).

An older upstream reference reported ITL p95 of approximately 1.1 s at a 64k
prefill versus 78 ms in steady decode. Those values describe a different
reference run, not the current 8192-token production observations above.

## Prefix caching

Prefix caching + preserved thinking (engine side) keep real agent operation in
the favourable scenario: after a tool call with 2k new tokens the whole 180k
session is not prefilled again. Reference measurement (20k prefix requested
twice): TTFT ~14 s → ~1.4 s.

## blazux reference — hybrid profile

Against unchanged NVFP4 on the same checkpoint family: ~+20% decode, ~8% more
KV space, agentic tournament 45/51 vs. 45/51 (quality unchanged). A controlled comparison for this deployment is still pending; our
production observations are documented separately.
