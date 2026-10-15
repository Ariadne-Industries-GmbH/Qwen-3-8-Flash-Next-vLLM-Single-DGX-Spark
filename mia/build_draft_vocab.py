#!/usr/bin/env python3
"""Fast equivalent of MiaAI files/build_draft_vocab.py with per-source
counters.

MiaAI's builder repeats each file N times for a ":N" weight and re-tokenizes
every pass. Here each file is tokenized exactly once and the per-source
counters are scaled by the weight -- mathematically identical counts, one
pass instead of ~13. Tokenization stays per chunk/line (1 MiB blocks / jsonl
lines) exactly like the original, so token boundaries are preserved.

  python3 mia/build_draft_vocab.py \
      corpus/english.txt \
      corpus/code.txt:3 \
      corpus/model_out.jsonl:20 \
      --model <local RadixArk checkpoint dir> \
      --size 65536 \
      --out mia/draft_vocab.txt \
      [--report-only]
Also prints the metric that matters for this deployment: coverage of the
model_out distribution alone by the selected 65k vocabulary (the drafter
predicts the model's own output).
"""
import argparse
import json
import sys
from collections import Counter

CHUNK = 1 << 20


def parse_weight(path: str) -> tuple[str, int]:
    if ":" in path and path.rsplit(":", 1)[1].isdigit():
        path, weight = path.rsplit(":", 1)
        return path, int(weight)
    return path, 1


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="draft_vocab.txt")
    ap.add_argument("--size", type=int, default=65536)
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
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for tid in ids:
                counts[tid] += 1
            docs += 1
            if docs % 500 == 0:
                print(f"  ... {docs} chunks, {sum(counts.values()):,} tokens",
                      file=sys.stderr, flush=True)
        per_source[path] = counts
        weights[path] = weight
        print(f"  {path}: {docs} docs, {sum(counts.values()):,} tokens, "
              f"{len(counts):,} distinct", file=sys.stderr, flush=True)

    def scenario(label: str, w: dict[str, int]) -> None:
        total: Counter = Counter()
        for path, counts in per_source.items():
            f = w.get(path, 1)
            if f == 1:
                total.update(counts)
            else:
                total.update({t: n * f for t, n in counts.items()})
        total_occ = sum(total.values())
        ranked = [t for t, _ in total.most_common()]
        keep = sorted(special)
        seen = set(keep)
        for tid in ranked:
            if len(keep) >= args.size:
                break
            if tid not in seen:
                keep.append(tid)
                seen.add(tid)
        covered = sum(total[t] for t in seen if t in total)
        print(f"\n=== scenario {label} (weights {w}) ===")
        print(f"corpus:      {total_occ:,} token occurrences, "
              f"{len(total):,} distinct ids")
        print(f"vocabulary:  {vocab_size:,} -> {len(keep):,} "
              f"({100.0 * len(keep) / vocab_size:.1f}%), "
              f"{len(special)} special/added kept unconditionally")
        print(f"coverage:    {100.0 * covered / total_occ:.4f}% of corpus occurrences")
        for cut in (8192, 16384, 32768, 65536, 131072):
            sub = special | set(ranked[:max(0, cut - len(special))])
            cov = sum(total[t] for t in sub if t in total)
            print(f"  size {cut:>7,}: coverage {100.0 * cov / total_occ:7.4f}%")
        # the metric that matters: coverage of the model output distribution
        out_counts = per_source.get(None, None)
        for path in per_source:
            if path.endswith("model_out.jsonl"):
                out_counts = per_source[path]
                break
        if out_counts:
            out_occ = sum(out_counts.values())
            out_cov = sum(out_counts[t] for t in seen if t in out_counts)
            print(f"model_out coverage by this vocabulary: "
                  f"{100.0 * out_cov / out_occ:.4f}% "
                  f"({sum(1 for t in out_counts if t in seen):,} of "
                  f"{len(out_counts):,} distinct ids)")
        return keep

    default_w = {p: weights[p] for p in per_source}
    keep_a = scenario("A: as given", default_w)

    # Scenario B: MiaAI's literal 20x output weight (reference only).
    w_b = dict(default_w)
    for p in w_b:
        if p.endswith("model_out.jsonl"):
            w_b[p] = 20
    if w_b != default_w:
        keep_b = scenario("B: MiaAI 20x output weight (reference)", w_b)
        same = set(keep_a) == set(keep_b)
        print(f"\nvocab A == vocab B: {same} "
              f"({len(set(keep_a) & set(keep_b)):,} shared ids)")

    if not args.report_only:
        with open(args.out, "w") as handle:
            handle.write("".join(f"{t}\n" for t in keep_a))
        print(f"\nwrote {len(keep_a):,} ids -> {args.out}")
        if w_b != default_w:
            out_b = args.out.rsplit(".", 1)[0] + "_mia20x_ref.txt"
            with open(out_b, "w") as handle:
                handle.write("".join(f"{t}\n" for t in keep_b))
            print(f"wrote {len(keep_b):,} ids -> {out_b} (reference only)")

if __name__ == "__main__":
    main()
