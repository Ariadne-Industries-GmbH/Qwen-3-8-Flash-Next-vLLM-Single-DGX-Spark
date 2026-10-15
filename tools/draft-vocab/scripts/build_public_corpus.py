#!/usr/bin/env python3
"""Builds the public draft-vocab corpus from the sources pinned in
sources.json. Writes corpus/<id>.txt and corpus/manifest.json.

  python3 scripts/build_public_corpus.py [--only english,german,code]
                                         [--cache-dir cache] [--out-dir corpus]
                                         [--max-mib N] [--local-file ID=PATH]

Download + extraction only, no AI, no training. Determinism: sources are
processed in sources.json order, read sequentially per source (no random
access), code files sorted by path and capped per source at max_mib. Same
sources.json => bit-identical corpus/<id>.txt.

Dependency: pyarrow only for hf_parquet sources (`uv pip install pyarrow`).
With --local-file an existing download can be reused (e.g. a wiki.train.raw
that is already stored locally) -- that path then does NOT end up in the
manifest, only the file hashes do.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
UA = {"User-Agent": "draft-vocab-corpus/1.0 (reproducible corpus build)"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, cache: Path, expect_sha: str = "") -> tuple[Path, str]:
    """Downloads (or uses the cache) and returns the file plus its sha256 -- the
    hash goes into the manifest so the build stays verifiable file by file."""
    name = url.rstrip("/").rsplit("/", 1)[-1] or "download"
    dst = cache / name
    if dst.is_file():
        digest = sha256_file(dst)
        if not expect_sha or digest == expect_sha:
            print(f"cache hit {name} ({dst.stat().st_size / 1e6:.0f} MB)")
            return dst, digest
        print(f"cache hash differs -> reloading: {name}")
        dst.unlink()
    cache.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    print(f"download {url}")
    req = urllib.request.Request(url, headers=UA)
    got = 0
    with urllib.request.urlopen(req) as resp, part.open("wb") as fh:
        for block in iter(lambda: resp.read(1 << 20), b""):
            fh.write(block)
            got += len(block)
            if got % (50 << 20) < CHUNK:
                print(f"  ... {got / 1e6:.0f} MB", flush=True)
    part.rename(dst)
    digest = sha256_file(dst)
    if expect_sha and digest != expect_sha:
        sys.exit(f"!! sha256 mismatch at {url}: {digest} != {expect_sha}")
    print(f"  {name}: {dst.stat().st_size / 1e6:.0f} MB sha256={digest[:16]}...")
    return dst, digest


class Budget:
    """Caps the amount of text written per source (MiB), so that the build
    stays deterministically bounded."""

    def __init__(self, max_mib: float):
        self.left = int(max_mib * 1024 * 1024)

    def take(self, fh, text: str) -> bool:
        if self.left <= 0:
            return False
        blob = text.encode("utf-8", "replace")[: self.left]
        fh.write(blob)
        self.left -= len(blob)
        return self.left > 0


def extract_hf_parquet(files: list[dict], cache: Path, repo: str, rev: str,
                       out, budget: Budget, downloads: dict) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("hf_parquet source requires pyarrow:  uv pip install pyarrow")
    for item in files:
        url = (f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{item['path']}")
        local, digest = fetch(url, cache, item.get("sha256", ""))
        downloads[url] = digest
        pf = pq.ParquetFile(str(local))
        col = item.get("column", "text")
        for rg in range(pf.num_row_groups):
            if budget.left <= 0:
                return
            table = pf.read_row_group(rg, columns=[col])
            for value in table.column(col).to_pylist():
                if value and not budget.take(out, value.rstrip("\n") + "\n"):
                    return


def strip_mediawiki(text: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", " ", text, flags=re.S)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    for _ in range(3):  # nested {{...}} peeled off approximately
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.S)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[='\"\[\]{}|*#;<>~_]", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def extract_stream_bz2(url: str, cache: Path, out, budget: Budget,
                       extractor: str, downloads: dict) -> None:
    """Reads the bz2 dump sequentially and stops as soon as the budget is full
    -- that way no more is downloaded than we actually need."""
    if extractor != "mediawiki_xml":
        sys.exit(f"unknown extractor {extractor}")
    # Read sequentially, so the file stays incomplete -> no file access, and
    # therefore no sha256 possible. The pointer in the manifest is the URL itself.
    downloads[url] = None
    req = urllib.request.Request(url, headers=UA)
    dec = bz2.BZ2Decompressor()
    pending = ""
    with urllib.request.urlopen(req) as resp:
        while budget.left > 0:
            raw = resp.read(1 << 20)
            if not raw:
                break
            pending += dec.decompress(raw).decode("utf-8", "replace")
            pages = pending.split("</page>")
            pending = pages.pop()
            for page in pages:
                m = re.search(r"<text[^>]*>(.*?)$", page, flags=re.S)
                if not m:
                    continue
                body = strip_mediawiki(m.group(1))
                if body and not budget.take(out, body + "\n"):
                    return


def extract_github_tarball(url: str, cache: Path, out, budget: Budget,
                           patterns: list[str], downloads: dict,
                           expect_sha: str = "") -> None:
    local, digest = fetch(url, cache, expect_sha)
    downloads[url] = digest
    matchers = [re.compile(fnmatch_to_regex(p)) for p in patterns]
    with tarfile.open(local, "r:gz") as tf:
        members = sorted(
            (m for m in tf.getmembers()
             if m.isfile() and any(rx.match(m.name) for rx in matchers)),
            key=lambda m: m.name)
        for m in members:
            if budget.left <= 0:
                return
            fh = tf.extractfile(m)
            if fh is None:
                continue
            body = fh.read().decode("utf-8", "replace")
            if not body.strip():
                continue
            budget.take(out, f"\n### {m.name}\n{body}\n")


def extract_local_file(path: Path, out, budget: Budget, downloads: dict) -> None:
    """Takes an already existing text file (option --local-file) up to the
    budget -- for dump files that are already sitting on disk."""
    if not path.is_file():
        sys.exit(f"--local-file: no such file: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while budget.left > 0:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            if not budget.take(out, chunk):
                break
    downloads[str(path.resolve())] = sha256_file(path)


def fnmatch_to_regex(pattern: str) -> str:
    return ".*" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(HERE / "sources.json"))
    ap.add_argument("--cache-dir", default=str(HERE / "cache"))
    ap.add_argument("--out-dir", default=str(HERE / "corpus"))
    ap.add_argument("--only", default="")
    ap.add_argument("--max-mib", type=float, default=0.0,
                    help="global cap; 0 = use the per-source value from sources.json")
    ap.add_argument("--local-file", action="append", default=[],
                    metavar="ID=PATH", help="existing file instead of a download")
    args = ap.parse_args()

    spec = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    locals_ = dict(kv.split("=", 1) for kv in args.local_file)

    manifest: dict[str, dict] = {}
    seen_files: dict[str, set[str]] = {}
    for src in spec["sources"]:
        sid = src["id"]
        if wanted and sid not in wanted:
            continue
        out_path = out_dir / f"{sid}.txt"
        writing = sid not in seen_files
        if writing:
            print(f"--- {sid}: {out_path}")
            out = out_path.open("wb")
        else:
            out = out_path.open("ab")
        seen_files.setdefault(sid, set())
        budget = Budget(args.max_mib or src.get("max_mib", 400))
        downloads: dict[str, str | None] = {}
        try:
            kind = src["kind"]
            if sid in locals_:
                extract_local_file(Path(locals_[sid]), out, budget, downloads)
            elif kind == "hf_parquet":
                extract_hf_parquet(src["files"], cache, src["repo"],
                                   src["repo_rev"], out, budget, downloads)
            elif kind == "wikimedia_dump":
                for item in src["files"]:
                    extract_stream_bz2(item["path"], cache, out, budget,
                                       item.get("extractor", ""), downloads)
            elif kind == "github_tarball":
                extract_github_tarball(src["url"], cache, out, budget,
                                       src.get("include", ["*"]), downloads,
                                       src.get("sha256", ""))
            else:
                sys.exit(f"unknown source-kind {kind}")
        finally:
            out.close()
        entry = manifest.setdefault(sid, {"files": [], "license": set(),
                                        "attribution": set(), "downloads": {}})
        entry["license"].add(src.get("license", "unknown"))
        entry["attribution"].add(src.get("attribution", ""))
        for f in src.get("files", []):
            p = f["path"] if isinstance(f, dict) else f
            entry["files"].append(p)
        # Tarball sources carry their address in "url", not in "files" --
        # without this the source would look empty in the manifest.
        if src.get("url"):
            entry["files"].append(src["url"])
        entry.setdefault("revs", []).append(src.get("repo_rev", ""))
        entry["downloads"].update(downloads)

    result = {}
    for sid, entry in manifest.items():
        path = out_dir / f"{sid}.txt"
        result[sid] = {
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256_file(path) if path.is_file() else "",
            "sources": entry["files"],
            "revs": sorted({r for r in entry["revs"] if r}),
            "license": sorted(entry["license"]),
            "attribution": sorted(a for a in entry["attribution"] if a),
            # null = streamed sequentially, so never hashed as a whole file;
            # otherwise the hash of the downloaded source file.
            "downloads": dict(sorted(entry["downloads"].items())),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
