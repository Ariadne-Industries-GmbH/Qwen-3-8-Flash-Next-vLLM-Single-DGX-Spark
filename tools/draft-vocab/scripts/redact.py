#!/usr/bin/env python3
"""Term replacement + hit check for corpus files.

Why this lives here: the reduced-vocab MTP profile works off a frequency
ranking of a text corpus. When the user adds their own model outputs to that
corpus (they match the drafter's distribution best), their content lands in the
corpus automatically. Before every build verify_corpus.py checks that no
identifying terms are left in it.

This script deliberately contains placeholders ONLY. The real rules come from
`redact_terms.json` (gitignored; template: redact_terms.example.json).

`forbidden_hits()` deliberately matches substrings instead of word boundaries:
`\bartikel\b` does not hit `artikel_modul`, and underscore identifiers are
exactly the shape project and system names take in logs and code.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_TERMS = Path(__file__).resolve().parents[1] / "redact_terms.json"

# Always active, independent of the user's file: the generic patterns.
# Private-key headers are an error in any corpus. The path and mail patterns,
# by contrast, are ordinary content in pinned public sources (code docs name
# home paths in the form /home/<user>/, a link list names a contact address,
# CPython file headers carry author addresses). They therefore count as a hard
# error only for OWN material -- that is what GENERIC_LABELS is for.
ALWAYS_FORBIDDEN: dict[str, re.Pattern[str]] = {
    "home_path": re.compile(r"/home/[a-z0-9._-]+/|/Users/[a-z0-9._-]+/"),
    "tilde_path": re.compile(r"~/[A-Za-z]"),
    "windows_user_path": re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]", re.I),
    "mail_address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+"),
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
}

# Labels of the patterns that may occur as content in published source
# material. A hit there is provenance noise, not a leak; in own material
# (model outputs, own code) it stays one.
GENERIC_LABELS: frozenset[str] = frozenset({
    "home_path", "tilde_path", "windows_user_path", "mail_address",
})


@lru_cache(maxsize=4)
def load_rules(terms_file: str | None = None) -> tuple[list[tuple[re.Pattern[str], str]], dict[str, re.Pattern[str]]]:
    """-> ([(pattern, replacement)], {label: forbidden_pattern})"""
    path = Path(terms_file) if terms_file else DEFAULT_TERMS
    rules: list[tuple[re.Pattern[str], str]] = []
    forbidden: dict[str, re.Pattern[str]] = dict(ALWAYS_FORBIDDEN)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("replace", []):
            needle = item["term"]
            # Substring, case-insensitive: covers Foo, foo, FOO and foo_bar.
            rules.append((re.compile(re.escape(needle), re.I), item.get("with", "REDACTED")))
        for item in data.get("forbid", []):
            forbidden[str(item)] = re.compile(re.escape(str(item)), re.I)
    else:
        print(f"NOTE: no terms file ({path.name}) -- only the generic patterns "
              f"are active. Own rules: create redact_terms.json from "
              f"redact_terms.example.json.", flush=True)
    return rules, forbidden


def redact(text: str, terms_file: str | None = None) -> str:
    rules, _ = load_rules(terms_file)
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def forbidden_hits(text: str, terms_file: str | None = None) -> list[str]:
    _, forbidden = load_rules(terms_file)
    return sorted({label for label, rx in forbidden.items() if rx.search(text)})


def selftest() -> int:
    """Regression guard for the failure modes that actually slipped through in
    an earlier redaction run:

    1. Word-boundary matching misses underscore identifiers (name_modul),
       because the underscore itself is a word character -> hence: substring
       matching is mandatory.
    2. Letter case and surname variants of one name -> hence: variants are
       mandatory per term.
    3. Windows paths also appeared twice-escaped -> the simple form must be
       hit as well.
    """
    import tempfile

    bs = "\\"  # one backslash, without spelling it out as a word in the source
    # The probes for the path/mail/key patterns are assembled from pieces,
    # built the same way as bs above, to avoid literal credential and user-path
    # patterns in the test source.
    with tempfile.TemporaryDirectory() as td:
        tf = Path(td) / "terms.json"
        tf.write_text(json.dumps({"replace": [{"term": "demo", "with": "[REDACTED]"}],
                                  "forbid": ["demo"]}), encoding="utf-8")

        for probe in ("demo_module", "DEMO Module", "see demo/dir"):
            out = redact(probe, str(tf))
            assert "demo" not in out.lower(), (probe, out)

        assert "demo" in forbidden_hits("x DEMOS_MODULE y", str(tf)), \
            "identifier variant with _ not detected"
        assert "home_path" in forbidden_hits(
            "see " + "/ho" + "me/" + "dev/x", str(tf))
        assert "mail_address" in forbidden_hits(
            "a@b." + "example.org", str(tf))
        assert "windows_user_path" in forbidden_hits(f"C:{bs}Users{bs}dev", str(tf))
        assert "windows_user_path" in forbidden_hits(f"C:{bs * 2}Users{bs * 2}dev", str(tf))
        assert "private_key" in forbidden_hits(
            "BEGIN RSA " + "PRIVATE KEY", str(tf))

    print("selftest ok: substring redaction, name variants, "
          "path/mail/key patterns")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    text = sys.stdin.read()
    sys.stdout.write(redact(text))
