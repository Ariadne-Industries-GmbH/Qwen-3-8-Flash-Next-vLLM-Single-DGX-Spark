#!/usr/bin/env python3
"""Gate for the draft-vocab artifact itself.

A vocab is a subset of the tokenizer vocabulary (one ID per line). It contains
no texts -- but every token that landed in the ranking ONLY because of private
material is a piece of information about exactly that material. An attacker can
decode the IDs back with the public tokenizer. This script turns precisely that
attack into the test.

  # 1) fast artifact check (decoding + patterns), seconds:
  python3 scripts/verify_vocab.py --model <ckpt> ../draft_vocab_public.txt

  # 2) provenance proof: every ID must occur in a public corpus
  #    (or be a tokenizer special). Costs tokenizing the
  #    corpora (~10-20 min), but it is the actual proof:
  python3 scripts/verify_vocab.py --model <ckpt> ../draft_vocab_public.txt \
      --corpora corpus/english.txt corpus/german.txt corpus/code.txt

Hit = exit 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import load_rules  # noqa: E402

CHUNK = 1 << 20


def load_ids(path: Path) -> list[int]:
    ids = [int(x) for x in path.read_text().split() if x.strip()]
    if not ids:
        sys.exit(f"empty vocab: {path}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab")
    ap.add_argument("--model", required=True)
    ap.add_argument("--terms", default=None)
    ap.add_argument("--corpora", nargs="*", default=[])
    ap.add_argument("--max-samples", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _, forbidden = load_rules(args.terms)
    ids = load_ids(Path(args.vocab))
    special = {int(i) for i in (tok.all_special_ids or [])}
    added = getattr(tok, "added_tokens_encoder", {}) or {}
    special |= {int(i) for i in added.values()}

    print(f"vocab {args.vocab}: {len(ids):,} ids "
          f"({len(set(ids)):,} unique), {len(special)} special/added")
    if len(set(ids)) != len(ids):
        print("WARNING: duplicate IDs", file=sys.stderr)

    bad: list[tuple[int, str, list[str]]] = []
    for i in ids:
        s = tok.decode([i])
        hits = [lab for lab, rx in forbidden.items() if rx.search(s)]
        if hits:
            bad.append((i, s, hits))
    if bad:
        print(f"\nARTIFACT LEAKS: {len(bad)} ID(s) with a hit pattern")
        for i, s, hits in bad[:args.max_samples]:
            print(f"  {i}: {s!r}  -> {', '.join(hits)}")
        return 1
    print("Decoded artifact check: no hit patterns.")

    if not args.corpora:
        print("\noptional: --corpora <public files> = "
              "provenance proof (every ID must occur there).")
        return 0

    allowed = set(special)
    for path in args.corpora:
        p = Path(path)
        if not p.is_file():
            print(f"MISSING: {p}", file=sys.stderr)
            return 2
        found: set[int] = set()
        if p.suffix == ".jsonl":
            import json
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        found |= set(tok(json.loads(line).get("text", ""),
                                         add_special_tokens=False)["input_ids"])
        else:
            with p.open(encoding="utf-8", errors="replace") as fh:
                while True:
                    block = fh.read(CHUNK)
                    if not block:
                        break
                    found |= set(tok(block, add_special_tokens=False)["input_ids"])
        print(f"{p.name}: {len(found):,} IDs from public corpus")
        allowed |= found

    unjustified = set(ids) - allowed
    if unjustified:
        print(f"\nPROVENANCE NOT PROVEN: {len(unjustified)} ID(s) in none "
              f"of the public corpora and not a special token.")
        for i in sorted(unjustified)[:args.max_samples]:
            print(f"  {i}: {tok.decode([i])!r}")
        print("-> These IDs come from material that was not listed in --corpora. "
              "Either add the corpus or remove the IDs.")
        return 1
    print("Provenance proof: every ID is covered by a public corpus "
          "or by a special token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
