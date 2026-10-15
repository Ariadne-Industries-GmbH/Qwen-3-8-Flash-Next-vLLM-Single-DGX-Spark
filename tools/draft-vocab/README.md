# Draft-vocab pipeline (reduced MTP draft vocabulary)

The `mia`/`hybrid` profile slices the `lm_head` of the MTP drafter down to a
subset of the tokenizer vocabulary (default 65,536 of 248,077 IDs). That
subset is a **frequency ranking of a text corpus** - no model, no training,
just counting. This directory holds the pipeline and the recipes; **the corpus
files themselves are not in the repo** (too large, licensing of the individual
sources, and for your own outputs they would carry personal data anyway).

```text
tools/draft-vocab/
  sources.json                     pinned public sources (+ license)
  redact_terms.example.json        template for your own replacement terms
  scripts/build_public_corpus.py   fetch sources + extract -> corpus/<id>.txt
  scripts/capture_model_outputs.py capture your own model outputs (optional)
  scripts/redact.py                term application + hit check (substring!)
  scripts/verify_corpus.py         hit check over the corpus files
  scripts/build_draft_vocab.py     tokenize, rank, write the vocab
  scripts/verify_vocab.py          gate over the artifact (decode it!)
  corpus/                          gitignored (the texts land here)
  cache/                           gitignored (downloads)
```

## Build

The normal route is the start script two directories up. It pins the source
revisions, the weights, the tokenizer path and both gate levels in one place,
and it is the same path customers will use later:

```bash
../../scripts/qwen38f-vllm.sh vocab-public   # public sources only  -> ../../mia/draft_vocab_public.txt
```

Public build, step by step:

```bash
uv venv .venv && uv pip install -r requirements.txt
.venv/bin/python scripts/build_public_corpus.py            # ~2-15 min download

# Gate 1 - the corpus. The mode belongs to the file, not to the run: public
# sources get --public, your own material gets --strict. Never mix both in one
# --strict run; that one aborts permanently over source noise (see below).
.venv/bin/python scripts/verify_corpus.py --public corpus/english.txt corpus/german.txt corpus/code.txt

.venv/bin/python scripts/build_draft_vocab.py \
    corpus/english.txt:1 corpus/german.txt:3 corpus/code.txt:3 \
    --model ../../../Qwen3.8-Flash-Next-NVFP4-RadixArk --size 65536 \
    --manifest corpus/manifest.json --out ../../mia/draft_vocab_public.txt \
    --provenance ../../mia/draft_vocab_public.txt.provenance.json

# Gate 2 - the artifact. This is the test that counts, and it runs in every
# build automatically; --corpora is the optional provenance proof.
.venv/bin/python scripts/verify_vocab.py \
    --model ../../../Qwen3.8-Flash-Next-NVFP4-RadixArk ../../mia/draft_vocab_public.txt \
    --corpora corpus/english.txt corpus/german.txt corpus/code.txt
```

`--provenance` is optional (default `<out>.provenance.json`). The proof for the
public vocab belongs in the repo without the `.local` suffix, because it is what
documents where that vocab came from; the proof of a private build names your
own sources and stays local. If the builder finds an absolute home path in a
proof, it aborts - the checkpoint belongs there as a name plus tokenizer hash,
not as a path.

The commands above use public inputs only. The command path is described
in [the public vocabulary documentation](../../docs/draft-vocab.md).

## What the artifact reveals about you - and how to measure it

A vocab is not a collection of texts, but it is a **subset**: which IDs ended up
in it depends on the distribution of the corpus. Anyone who decodes the artifact
with the public tokenizer sees the vocabulary the corpus was written in. The
test for that is `verify_vocab.py`:

* **fast** (seconds): decode every ID, match it against the patterns from
  `redact_terms.json`. A single hit = exit 1.
* **provenance proof** (~10-20 min): every ID must occur in one of the
  `--corpora` that were passed, or be a special token. IDs that come only from
  material that was left out are listed by name.

The provenance proof is the reason your own outputs are a deliberate step in the
corpus and not a default: without them the artifact is documented by the source
list, with them only by the redaction.

Rule of thumb from building the shipped public vocab: the vocabulary of pure
message corpora is strongly shaped by the **language** of operation. A
deployment that mostly answers in German needs a German branch in the corpus
(hence `german` in `sources.json`) - otherwise a large share of the German
tokens is missing from the draft head and the acceptance rate drops while the
output stays identical.

## Licenses of the sources (see sources.json)

| Source | License |
|---|---|
| WikiText-103 raw train | CC BY-SA 3.0 / GFDL (dataset card) |
| German Wikipedia article dump (date pinned) | CC BY-SA 4.0 |
| vLLM, Transformers | Apache-2.0 |
| CPython | PSF-2.0 (parts GPL-3.0+) |

We distribute **no** source texts of these projects - we only use them as input
for a frequency count. When passing on the resulting `draft_vocab_public.txt`,
please include the CC-BY-SA attribution (WikiText, Wikipedia), as the license
requires for derived collections.
