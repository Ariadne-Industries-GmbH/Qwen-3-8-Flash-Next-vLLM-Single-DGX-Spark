# Quick start (customer path)

Everything below is one path: **download the model, build the image, start.**
No vocabularies to build, no calibration runs, no patching, no hand-edited
config. Configuration lives in `README.md`; measurements live in `docs/`; this file only contains
the steps.

Target hardware: **DGX Spark (GB10, ~121.7 GiB unified memory)**. The serving
configuration is tuned for that pool and for the PLE shards living on its NVMe;
on other machines the memory numbers in `README.md` do not transfer.

Prerequisites on that machine: Docker with the compose v2 plugin, `git`,
`python3`, `curl`, `sudo` for the one cache-clearing command, and the
Hugging Face CLI (`pip install "huggingface_hub[cli]"`) for the download.
Plan for ~135 GB of disk plus ~8 GB more if you use `hybrid`; the checkpoint
should sit on the machine's NVMe, since the PLE tables are read from it at
random during decode.

---

## Build and start

```bash
# Run from this repository directory.
# 1) Download the model into the sibling checkpoint directory (~135 GB).
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir ../Qwen3.8-Flash-Next-NVFP4-RadixArk

# 2) Images, once (pulls a ~10.5 GB base, then builds the patch layer on top)
./scripts/qwen38f-vllm.sh build-image
./scripts/qwen38f-vllm.sh build-image-mia

# 3) Prepare the production checkpoint (once, all servers stopped)
./scripts/qwen38f-vllm.sh prepare-hybrid

# 4) Start
sudo ./scripts/qwen38f-vllm.sh dropcaches
./scripts/qwen38f-vllm.sh hybrid
```

`build-image` clones and pins the build repository itself
(`blazux/qwen3.8-Flash-DGX` @ `656cdb075b57671c9bd6807b57f630fb3c1641fe`) and
checks the commit afterwards. Nothing has to be cloned by hand.

**First boot can take 10-15 minutes** until `/health` answers, depending on
the image, profile and cache state. That is normal, not a hang. Watch the progress
with `./scripts/qwen38f-vllm.sh logs hybrid`.

---

## Two commands to confirm it works

```bash
./scripts/qwen38f-vllm.sh kv hybrid # KV size + "MTP draft vocab: 65536 of 248077"
./scripts/qwen38f-vllm.sh test    # readiness check + one real completion
```

The launcher follows the container log after starting. Use a second terminal
for verification, or press Ctrl+C to stop following logs; the detached server
keeps running.

Acceptance criteria:

| Check | Expected |
|---|---|
| `kv` | `GPU KV cache size` at least **720k tokens**, `Maximum concurrency ... >= 4.00x` |
| `kv` | `MTP draft vocab: 65536 of 248077` (the shipped vocabulary is active) |
| `logs hybrid` | no `RuntimeError` / `ValueError`, `Application startup complete` present |
| `test` | HTTP 200 with a completed answer |
| endpoint | `http://127.0.0.1:44415/v1/models` -> `qwen3.8-flash-next` |

---

## Already inside this repo -- nothing to generate

| Item | State |
|---|---|
| Draft vocabulary `mia/draft_vocab_public.txt` | **included**, 65,536 IDs, built only from public sources |
| Its provenance record | `mia/draft_vocab_public.txt.provenance.json` (source URLs, revisions, SHA-256, licenses) |
| PLE prefetch patch (`ple_patches/`) | **included**; no extraction from the image is needed for this patch |
| PLE `MADV_RANDOM` + MTP vocab patch | baked into the image by `build-image-mia` |
| Startup checks (model, images, vocab, patches, RAM) | run automatically before every start |
| Tests | `tests/` |

The launcher uses the shipped public vocabulary by default. No vocabulary
build is needed for startup. `vocab-public` rebuilds it from the pinned public
sources; alternative files require an explicit `DRAFT_VOCAB_FILE` override.

The public vocabulary is built exclusively from: WikiText-103 (English), the
German Wikipedia article dump, and the pinned source archives of vLLM,
Transformers and CPython. Weights `english:1 german:3 code:3`. No customer
data, no private corpus.

---

## Recommended setting: the `hybrid` profile

The production path above uses **`hybrid`**, which converts the dense side
layers of your checkpoint to FP8 locally: about **7 GiB fewer resident
weights**, which flow straight into the PLE page cache (roughly +20 % decode in
the reference measurement). To prepare or rebuild it:

```bash
./scripts/qwen38f-vllm.sh down
./scripts/qwen38f-vllm.sh prepare-hybrid     # once, ~10 min, CPU only
./scripts/qwen38f-vllm.sh hybrid
```

The converted checkpoint is generated locally rather than shipped in this
repository. `prepare-hybrid` derives it from the
original RadixArk download: a hardlink clone of your own copy, ~7.7 GiB of
genuinely new bytes, original left untouched (the driver proves that
afterwards). The server must be down during the conversion.

Stop with `./scripts/qwen38f-vllm.sh down`, then roll back with
`./scripts/qwen38f-vllm.sh k3` -- k3 and mia serve
the unmodified checkpoint. Delete the snapshot with
`rm -rf ../Qwen3.8-Flash-Next-NVFP4-RadixArk-fp8hybrid`.

`mia` adds reduced-vocabulary drafting to `k3`; `hybrid` additionally uses
FP8 side layers. All three profiles share port,
context length and KV budget.

---

## Configuration

Defaults are provided by the launcher and Compose, so a `.env` file is optional.
`cp .env.example .env` if you want them explicit. Variables you export in the
shell always win over `.env`.

The ones worth knowing:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `../Qwen3.8-Flash-Next-NVFP4-RadixArk` | checkpoint location |
| `VLLM_PORT` | `44415` | OpenAI-compatible API |
| `QFN_MAX_MODEL_LEN` | `180000` | context per request |
| `QFN_MAX_NUM_SEQS` | `4` | parallel requests |
| `QFN_MTP_NUM_SPECULATIVE_TOKENS` | `3` | MTP draft length (sweep optimum) |
| `QFN_KV_CACHE_MEMORY_BYTES` | `12459440000` | hard KV budget in bytes |
| `QFN_MAX_NUM_BATCHED_TOKENS` | `8192` | shared scheduler budget |
| `QFN_MIA_MAX_NUM_BATCHED_TOKENS` | `8192` | Mia/hybrid scheduler budget |
| `RESTART_POLICY` | `no` | `unless-stopped` to survive a reboot |

Start parameters can also be overridden per call, e.g.
`QFN_MAX_MODEL_LEN=... ./scripts/qwen38f-vllm.sh mia`.

---

## When something does not start

| Symptom | Cause and fix |
|---|---|
| Aborts with a MemAvailable / pool message | Weight leftovers or another large process occupy the pool: `sudo ./scripts/qwen38f-vllm.sh dropcaches` and stop other large consumers, then start again |
| `/health` silent for >20 min | `./scripts/qwen38f-vllm.sh logs hybrid` -- first boot is slow, an error would be in the log |
| Launcher asks for a draft vocab | The repo copy is missing from your checkout; `git checkout mia/draft_vocab_public.txt` |
| Cache folders appear at `/` root | The start ran without `HOME` set; start from a normal shell or set the `*_CACHE_DIR` variables in `.env` |
| Need the exact memory reasoning | [Memory management](docs/memory-management.md) |

Optional host tuning (kernel reclaim watermarks, needs root, only while no
server runs): `sudo ./scripts/host-tuning.sh status` first, `apply` only if the
numbers suggest it. [Memory management](docs/memory-management.md) explains the effects.

## Housekeeping

```bash
./scripts/qwen38f-vllm.sh down         # stop
./scripts/qwen38f-vllm.sh ps hybrid       # container state
./scripts/qwen38f-vllm.sh logs hybrid     # follow the log
```

`LICENSE` (AGPL-3.0-or-later) applies to this repository and must stay with it;
See the [license section](README.md#license).
