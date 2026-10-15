#!/usr/bin/env python3
"""Capture your own model outputs as a vocab source (optional, recommended).

Why bother at all: the MTP drafter has to predict the output of the target
model. A corpus made of your own model outputs matches exactly that
distribution best (MiaAI recipe: high weight on model_out). But it is also the
only part of the corpus that CAN contain personal material. Hence: only what has
passed the redaction comes out of here, and the rest stays on your own machine.

  python3 scripts/capture_model_outputs.py --in export.jsonl --out corpus/model_out.jsonl

Input formats (--in can be a file or a directory):
  *.jsonl  one object per line; recognized fields: role|is_ai_speaking,
           content|text, reasoning_content|reasoning, tool_calls
  *.txt    complete text, roles are ignored (fallback only)

Reasoning markers are restored in native form (the marker sequence is built
here via chr(), so this file itself does not look like a chat template): at
runtime the drafter sees exactly this stream.

Hook up your own database: 20 lines in your own adapter, then use it like --in.
Pattern (the SQL is deliberately kept generic):

  SELECT content, reasoning_content, tool_calls FROM messages
   WHERE is_ai = 1 AND created_at >= :since

Output: deduplicated, redacted, one JSON line per document.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import forbidden_hits, redact  # noqa: E402

_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/" + "think" + chr(62)


def build_text(rec: dict) -> str:
    parts = []
    reasoning = (rec.get("reasoning_content") or rec.get("reasoning") or "").strip()
    content = (rec.get("content") or rec.get("text") or "").strip()
    if reasoning:
        parts.append(f"{_THINK_OPEN}{reasoning}{_THINK_CLOSE}")
    if content:
        parts.append(content)
    tool_calls = (rec.get("tool_calls") or "").strip()
    if tool_calls and tool_calls not in ("[]", "null"):
        parts.append(tool_calls)
    return "\n".join(parts).strip()


def is_ai(rec: dict) -> bool:
    if "is_ai_speaking" in rec:
        return bool(rec["is_ai_speaking"])
    role = str(rec.get("role", "")).lower()
    return role in ("assistant", "ai", "model")


def iter_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        yield {"role": "assistant", "text": path.read_text(encoding="utf-8",
                                                           errors="replace")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="",
                    help="ISO time cutoff, if the source carries a created_at field")
    ap.add_argument("--strict", action="store_true",
                    help="Drop documents with pattern hits instead of only reporting them")
    ap.add_argument("--terms", default=None)
    args = ap.parse_args()

    src = Path(args.src)
    paths = sorted(src.rglob("*.jsonl")) if src.is_dir() else [src]
    if not paths:
        print(f"no input files in {src}", file=sys.stderr)
        return 2
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) \
        if args.since else None

    seen: set[str] = set()
    kept = dropped_ai = skipped = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            for rec in iter_records(path):
                if since:
                    ts = rec.get("created_at") or rec.get("timestamp")
                    if ts:
                        try:
                            if datetime.fromisoformat(str(ts).replace("Z", "+00:00")) < since:
                                continue
                        except ValueError:
                            pass
                if not is_ai(rec):
                    continue
                text = build_text(rec)
                if not text or text in seen:
                    continue
                seen.add(text)
                redacted = redact(text, args.terms)
                hits = forbidden_hits(redacted, args.terms)
                if hits and args.strict:
                    skipped += 1
                    continue
                out.write(json.dumps({"text": redacted}, ensure_ascii=False) + "\n")
                kept += 1
                if hits:
                    dropped_ai += 1
    print(f"sources:        {len(paths)}")
    print(f"written:        {kept} documents -> {out_path}")
    if dropped_ai:
        print(f"hits present:     {dropped_ai} (without --strict; check the terms)")
    if skipped:
        print(f"discarded:      {skipped} (--strict)")
    return 0 if not dropped_ai else 1


if __name__ == "__main__":
    raise SystemExit(main())
