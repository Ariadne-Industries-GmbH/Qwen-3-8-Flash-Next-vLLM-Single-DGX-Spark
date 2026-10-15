# Mia profile: reduced-vocabulary MTP drafting

The Mia image adds a reduced MTP draft head, explicit CUDA graph capture widths
`[4,8,12,16]`, the V2 model runner and PLE `MADV_RANDOM` to the base image.
The production `hybrid` profile uses this image with FP8 side layers.
All vLLM profiles use an 8192-token shared scheduler budget.

## Vocabulary

The default input is `draft_vocab_public.txt`: 65,536 unique token IDs built
from WikiText-103, German Wikipedia, and pinned vLLM, Transformers and CPython
source archives. Weights are English 1, German 3, code 3. The accompanying
`.provenance.json` records source revisions, hashes and attribution.
The container mounts the file read-only at `/root/draft_vocab.txt`.

No vocabulary generation is required to start. Rebuild the public input with
`./scripts/qwen38f-vllm.sh vocab-public` from the repository root.
See [data provenance](../docs/draft-vocab.md) and the
[pipeline instructions](../tools/draft-vocab/README.md).

## Build and validation

`./scripts/qwen38f-vllm.sh build-image-mia` generates `mtp_patched.py` and
`vllm_ple_mmap_patched.py` plus their `.orig` intermediates. These generated
files are excluded from publication. The patch scripts and `Dockerfile` are
versioned inputs and must remain available.

The runtime prefetch module in `../ple_patches/` is a versioned dependency
of the hybrid profile, not a disposable build intermediate.

See [profile mechanics and A/B measurements](../docs/mia-profile.md) and
[performance references](../docs/performance-reference.md).
