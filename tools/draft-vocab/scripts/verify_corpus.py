#!/usr/bin/env python3
"""Hit check over corpus files: whatever contains identifiable material does
not stay on the publication path.

  python3 scripts/verify_corpus.py --strict corpus/*.jsonl corpus/*.txt
  python3 scripts/verify_corpus.py --public corpus/english.txt corpus/code.txt

Rules: tools/draft-vocab/redact_terms.json (template: .example.json) plus the
generic patterns (home paths, mail addresses, private-key headers). The match is
substring + case-insensitive -- word-boundary regexes on underscore identifiers
are the classic redaction trap.

Three modes:  --strict  hard error on every hit (for own material,
                     e.g. model outputs; default in the build wrapper)
            --public  hits from the terms file stay fatal; hits from the
                     generic path/mail patterns are ordinary content in pinned
                     public sources and are only reported, without blocking the
                     build
            (neither)  everything as a note, exit 0
The hard gate for a publication vocab stays, in every mode, the artifact check
with provenance proof (verify_vocab.py --corpora).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import (GENERIC_LABELS, forbidden_hits, load_rules,  # noqa: E402
                    redact)


def iter_texts(path: Path):
    if path.suffix == ".jsonl":
        import json
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line).get("text", "")
    else:
        with path.open(encoding="utf-8", errors="replace") as fh:
            while True:
                block = fh.read(1 << 20)
                if not block:
                    break
                yield block


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    public = "--public" in sys.argv[1:]
    argv = [a for a in sys.argv[1:] if a not in ("--strict", "--public")]
    terms = None
    paths = []
    for arg in argv:
        if arg.startswith("--terms="):
            terms = arg.split("=", 1)[1]
        else:
            paths.append(Path(arg))
    if not paths:
        print(__doc__)
        return 2

    rules, forbidden = load_rules(terms)
    print(f"rules: {len(rules)} replacements, {len(forbidden)} hit patterns")
    total: Counter[str] = Counter()
    bad_docs = 0
    docs = 0
    for path in paths:
        if not path.is_file():
            print(f"MISSING: {path}", file=sys.stderr)
            return 2
        n = 0
        for text in iter_texts(path):
            docs += 1
            n += 1
            hits = forbidden_hits(text, terms)
            # Check again after the replacement: a rule can itself create a
            # term (rare) or a remainder that cannot be replaced.
            hits += [h for h in forbidden_hits(redact(text, terms), terms)
                     if h not in hits]
            for h in set(hits):
                total[h] += 1
            if hits:
                bad_docs += 1
        print(f"{path}: {n} document blocks")
    print(f"\ndocuments: {docs}, with hits: {bad_docs}")
    if not total:
        print("CORPUS CLEAN: no hit pattern matched.")
        return 0
    print("Pattern hits:")
    for label, count in total.most_common():
        art = "  [generic pattern]" if label in GENERIC_LABELS else ""
        print(f"  {label}: {count} document(s){art}")
    private = {k: v for k, v in total.items() if k not in GENERIC_LABELS}
    generic = {k: v for k, v in total.items() if k in GENERIC_LABELS}
    if private:
        print(f"\nERROR: {sorted(private)} come from the terms file -- this is "
              "a leak indicator regardless of the mode. Extend the redaction "
              "or exclude the affected documents.")
        return 1
    if generic and strict:
        print("\n--strict: exit 1. Either add/extend the terms "
              "or exclude the affected documents.")
        return 1
    if generic and public:
        print("\nonly generic patterns matched -- expected content in the pinned "
              "public sources (revision/hash: sources.json). "
              "The hard gate is and stays the artifact check with the "
              "provenance proof: verify_vocab.py --corpora ...")
        return 0
    print("\n(note: --strict turns this into a hard error, --public "
          "rates the generic patterns as source noise.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
