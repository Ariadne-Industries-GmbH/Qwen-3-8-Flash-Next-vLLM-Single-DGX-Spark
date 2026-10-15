#!/usr/bin/env python3
"""Draft-vocab builder: frequency ranking of the corpus -> the IDs the MTP
drafter is allowed to read.

  python3 scripts/build_draft_vocab.py \
      corpus/english.txt \
      corpus/german.txt \
      corpus/code.txt:3 \
      corpus/model_out.jsonl:20 \
      --model <local checkpoint dir> \
      --size 65536 \
      --out ../draft_vocab_public.txt \
      [--manifest corpus/manifest.json] [--report-only]

Weights are counter scaling, not repetition: each file is tokenized exactly
once and the counters are multiplied by the weight. That is mathematically
identical to "run the file N times" but saves ~13 passes. Tokenization stays
per chunk/line (1 MiB / jsonl line) as in the original, so token boundaries
stay the same.

On top of that: coverage of the model-output corpus shown separately (that is
the distribution the drafter has to predict) and a size ladder 8k..131k.

With --manifest the provenance record is written (--provenance, default
<out>.provenance.json): which sources with which file hash went into this
artifact. That is the record that later shows where a vocab came from.

The record for the public vocab is published along with it, so it must contain
NO local user path: the checkpoint is noted as a name plus a tokenizer hash,
not as a path, and the write aborts with a message if an absolute home path
shows up in the JSON anyway. Pass the corpus files relative to the tool
directory (corpus/...), the
way the start script does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

CHUNK = 1 << 20

# Guard over the provenance JSON: a host home path in a published record is a
# leak, no matter how careful the arguments passed in were. Both common forms
# (Linux, macOS).
HOME_PATH = re.compile(r"/(?:[Hh]ome|[Uu]sers)/[A-Za-z0-9._-]+/")


def parse_weight(spec: str) -> tuple[str, int]:
    if ":" in spec and spec.rsplit(":", 1)[1].isdigit():
        path, weight = spec.rsplit(":", 1)
        return path, int(weight)
    return spec, 1


def iter_chunks(path: str):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line).get("text", "")
    else:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while True:
                block = handle.read(CHUNK)
                if not block:
                    break
                yield block


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_identity(model_dir: str) -> dict:
    """Checkpoint identity without a local path.

    The start script passes MODEL_DIR resolved -- an absolute path through the
    home directory. In a record that ships with the release that would be a
    leak of the host user. The file name of the checkpoint folder plus the
    hashes of its tokenizer files identify the same checkpoint on every other
    machine as well, without naming a path.
    """
    root = Path(model_dir)
    ident: dict = {"name": root.resolve().name}
    hashes: dict[str, str] = {}
    for fname in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        f = root / fname
        if f.is_file():
            hashes[fname] = sha256_file(str(f))
    if hashes:
        ident["tokenizer_sha256"] = hashes
    return ident


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="draft_vocab.txt")
    ap.add_argument("--provenance", default="",
                    help="Path of the provenance proof "
                         "(default: <out>.provenance.json)")
    ap.add_argument("--size", type=int, default=65536)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = len(tok)
    special = set(tok.all_special_ids or [])
    added = getattr(tok, "added_tokens_encoder", {}) or {}
    special |= {int(i) for i in added.values()}
    special = {i for i in special if 0 <= i < vocab_size}

    per_source: dict[str, Counter] = {}
    weights: dict[str, int] = {}
    for spec in args.corpus:
        path, weight = parse_weight(spec)
        print(f"tokenizing {path} (weight {weight}) ...", file=sys.stderr, flush=True)
        counts: Counter = Counter()
        docs = 0
        for text in iter_chunks(path):
            if not text:
                continue
            for tid in tok(text, add_special_tokens=False)["input_ids"]:
                counts[tid] += 1
            docs += 1
            if docs % 500 == 0:
                print(f"  ... {docs} chunks, {sum(counts.values()):,} tokens",
                      file=sys.stderr, flush=True)
        per_source[path] = counts
        weights[path] = weight
        print(f"  {path}: {docs} docs, {sum(counts.values()):,} tokens, "
              f"{len(counts):,} distinct", file=sys.stderr, flush=True)

    def select(w: dict[str, int]) -> tuple[list[int], Counter]:
        total: Counter = Counter()
        for path, counts in per_source.items():
            f = w.get(path, 1)
            total.update(counts if f == 1 else {t: n * f for t, n in counts.items()})
        ranked = [t for t, _ in total.most_common()]
        keep = sorted(special)
        seen = set(keep)
        for tid in ranked:
            if len(keep) >= args.size:
                break
            if tid not in seen:
                keep.append(tid)
                seen.add(tid)
        return keep, total

    def report(label: str, w: dict[str, int]) -> tuple[list[int], set[int]]:
        keep, total = select(w)
        seen = set(keep)
        total_occ = sum(total.values())
        covered = sum(total[t] for t in seen if t in total)
        print(f"\n=== {label} (weights {w}) ===")
        print(f"corpus:      {total_occ:,} token occurrences, {len(total):,} distinct ids")
        print(f"vocabulary:  {vocab_size:,} -> {len(keep):,} "
              f"({100.0 * len(keep) / vocab_size:.1f}%), "
              f"{len(special)} special/added always included")
        print(f"coverage:    {100.0 * covered / total_occ:.4f}% of corpus occurrences")
        for cut in (8192, 16384, 32768, 65536, 131072):
            sub = special | set(ranked_cuts(total, cut, len(special)))
            cov = sum(total[t] for t in sub if t in total)
            print(f"  size {cut:>7,}: coverage {100.0 * cov / total_occ:7.4f}%")
        for path in per_source:
            if path.endswith("model_out.jsonl"):
                oc = per_source[path]
                occ = sum(oc.values())
                cov = sum(oc[t] for t in seen if t in oc)
                print(f"model_out coverage: {100.0 * cov / occ:.4f}% "
                      f"({sum(1 for t in oc if t in seen):,} of {len(oc):,} distinct)")
                break
        return keep, seen

    def ranked_cuts(total: Counter, cut: int, n_special: int):
        return [t for t, _ in total.most_common()][:max(0, cut - n_special)]

    keep, seen = report("A: exactly as specified", {p: weights[p] for p in per_source})

    if not args.report_only:
        with open(args.out, "w") as handle:
            handle.write("".join(f"{t}\n" for t in keep))
        print(f"\nwrote {len(keep):,} ids -> {args.out}")
        prov = {
            "artifact_sha256": sha256_file(args.out),
            "size": len(keep),
            "vocab_size": vocab_size,
            "weights": weights,
            "model": model_identity(args.model),
            "sources": {p: {"sha256": sha256_file(p), "weight": w,
                           "distinct_ids": len(per_source[p])}
                        for p, w in weights.items()},
        }
        if args.manifest and Path(args.manifest).is_file():
            prov["corpus_manifest"] = json.loads(
                Path(args.manifest).read_text(encoding="utf-8"))
        blob = json.dumps(prov, indent=2, sort_keys=True)
        prov_path = args.provenance or args.out + ".provenance.json"
        leak = HOME_PATH.search(blob)
        if leak:
            sys.exit(f"!! The provenance record pins a local user path "
                     f"({leak.group(0)}...) -- unfitting as a publication proof. "
                     "Pass the corpus files relative to the "
                     "tool directory (corpus/.../: this is how "
                     "scripts/qwen38f-vllm.sh calls this script).")
        Path(prov_path).write_text(blob, encoding="utf-8")
        print(f"wrote provenance -> {prov_path}")


if __name__ == "__main__":
    main()
