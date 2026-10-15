# Public draft vocabulary

Mia and hybrid use `mia/draft_vocab_public.txt` by default, built only from
the public sources pinned in `tools/draft-vocab/sources.json`. The release
includes this ID list and its provenance record. Rebuild with
`./scripts/qwen38f-vllm.sh vocab-public`; no rebuild is required to serve.

## The public vocab

- **65,536 IDs**, built from:
  - **WikiText-103** (English raw)
  - **German Wikipedia** (dewiki dump)
  - **vLLM / Transformers / CPython** source code
- Weights `english:1 german:3 code:3` (overrides `DRAFT_VOCAB_*`, see `usage`),
  tokenizer from the local RadixArk checkpoint.
- Measured corpus coverage: **99.72%** of all token occurrences in the three
  public corpora; **121,414 distinct IDs** in the corpora.

The launch script selects the public vocab by default — same patch, same number of IDs, just a different subset of the
same vocabulary. The output is unaffected by this — the patch only changes
which candidates the drafter checks, not the distribution from which tokens
are ultimately accepted. The acceptance rate can be lower, when the serving workload differs from the public
corpus distribution; **this has not been measured
on this build** (a controlled comparison on the same hardware is required).

## The two gate levels

What ships are token IDs, not texts. Anyone decoding the artifact back with
the public tokenizer still sees the vocabulary the corpus was written in.
Hence two gate levels, and the mode belongs to the file in each case, not to
the run:

- **Corpus gate** (`verify_corpus.py`): public sources run with `--public`,
  because paths and e-mail addresses are *content* there — documentation names
  home paths in the form `/home/<user>/`, weblink lists a contact address,
  CPython file headers carry author addresses. Own material runs with
  `--strict`; there, every finding stays an abort. A shared `--strict` run
  across both kinds would not be a gate but a permanent build abort.
- **Artifact gate** (`verify_vocab.py`): every ID is decoded back and checked
  against the finding patterns — this is the test that counts, because IDs
  are what ships. Optional `--corpora <public files>`: **provenance proof**,
  every ID must occur in a public corpus or be a special token (~10-20 min).
  Only the artifact check runs automatically in every build. The previous build recorded the following counts for
  `mia/draft_vocab_public.txt`: 79,513
  (english) / 97,089 (german) / 44,865 (code) IDs from the public corpora, so
  the previous verification reported every shipped ID as covered.
  Membership alone
  does not establish that frequency ranking used exclusively public inputs;
  an exact rebuild is the stronger check.

## Provenance

The evidence file `mia/draft_vocab_public.txt.provenance.json` is part of the
repo: it lists the sources, revisions and file hashes of the sources that went
into the artifact. The checkpoint appears in it as name + tokenizer hash, not
as a path; if the builder nonetheless finds an absolute home path in the
evidence file, it aborts. Local overrides and their provenance are excluded from publication.

## Licenses

The [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
lists **CC BY-SA 3.0 and GFDL**; the dewiki dump is recorded as **CC BY-SA 4.0**, vLLM and Transformers
**Apache-2.0**, CPython **PSF-2.0** (parts GPL-3.0+). We do not redistribute
any source texts, we count frequencies over them. When passing the vocab on,
include the CC-BY-SA attribution (WikiText, Wikipedia).

Recipes and scripts: `tools/draft-vocab/README.md`.
