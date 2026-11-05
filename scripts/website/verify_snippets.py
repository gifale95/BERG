#!/usr/bin/env python3
"""
Run the usage snippets from models_overlay.json (and a quickstart block on the
homepage, if it has one) against the installed BERG package and local model files.

Each snippet runs with `berg` and `np` defined, plus the inputs named in its
`snippet_inputs` (synthetic arrays or strings) and, for `snippet_files`, a temporary
working directory where the named file links to --video. A snippet passes when it
finishes without error and defines `responses`.

With --write, passing snippets are marked verified in the overlay by storing a short
hash of the snippet text; failing ones are reset to false. build_catalog.py only
publishes a snippet whose current text matches that hash, so editing a snippet
un-verifies it until this script is run again.

Usage
-----
    python scripts/website/verify_snippets.py --berg-dir /path/to/brain-encoding-response-generator
    python scripts/website/verify_snippets.py --berg-dir ... --video clip.mp4 --write
    python scripts/website/verify_snippets.py --berg-dir ... --only eeg-things_eeg_2-vit_b_32
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "scripts" / "website" / "models_overlay.json"
INDEX_HTML = REPO / "docs" / "index.html"
QUICKSTART_PATH = "/path/to/brain-encoding-response-generator"


def snippet_hash(snippet: str) -> str:
    return hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:12]


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "rgb_images": lambda: rng.integers(0, 256, (2, 3, 224, 224), dtype=np.uint8),
        "rgb_videos": lambda: rng.integers(0, 256, (1, 16, 3, 256, 256), dtype=np.uint8),
        "gray_videos": lambda: rng.integers(0, 256, (1, 60, 144, 256), dtype=np.uint8),
        "tr_transcripts": lambda: ["Hello, are you", "awake? Yes,", "I am here."],
    }


def quickstart_snippet() -> str | None:
    page = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r'<pre[^>]*data-verify="quickstart"[^>]*><code[^>]*>(.*?)</code></pre>', page, re.S)
    return html.unescape(match.group(1)) if match else None


def describe(value) -> str:
    if hasattr(value, "shape"):
        return f"shape {tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} of {len(value)}"
    if isinstance(value, dict):
        return f"dict with keys {list(value)[:4]}"
    return type(value).__name__


def run_snippet(code, berg, inputs, files, berg_dir):
    namespace = {"berg": berg, "np": np, **inputs}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        for name, source in files.items():
            os.symlink(os.path.abspath(source), os.path.join(tmp, name))
        os.chdir(tmp)
        try:
            exec(compile(code.replace(QUICKSTART_PATH, berg_dir), "<snippet>", "exec"), namespace)
        finally:
            os.chdir(cwd)
    if "responses" not in namespace:
        raise RuntimeError("snippet did not define `responses`")
    return describe(namespace["responses"])


def set_verified(text: str, model_id: str, value: str | None) -> str:
    start = text.index(f'"{model_id}": {{')
    match = re.compile(r'"snippet_verified": (?:false|true|"[0-9a-f]*")').search(text, start)
    new = f'"snippet_verified": "{value}"' if value else '"snippet_verified": false'
    return text[: match.start()] + new + text[match.end():]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--berg-dir", required=True)
    p.add_argument("--video", help="a short video file, needed by snippets that take a file path")
    p.add_argument("--only", nargs="+", help="model ids to run (default: all with a snippet)")
    p.add_argument("--skip-quickstart", action="store_true")
    p.add_argument("--write", action="store_true", help="record results in models_overlay.json")
    p.add_argument("--traceback", action="store_true", help="print full tracebacks for failures")
    args = p.parse_args()

    from berg import BERG
    berg = BERG(args.berg_dir)
    overlay_text = OVERLAY.read_text(encoding="utf-8")
    models = json.loads(overlay_text)["models"]
    factories = make_inputs()
    file_sources = {"video": args.video}

    jobs = []
    if not args.skip_quickstart and not args.only:
        jobs.append(("quickstart (index.html)", quickstart_snippet(), {}, {}))
    for model_id, entry in models.items():
        if args.only and model_id not in args.only:
            continue
        if entry.get("snippet"):
            jobs.append((model_id, entry["snippet"], entry.get("snippet_inputs", {}), entry.get("snippet_files", {})))

    results = []
    for name, code, input_spec, file_spec in jobs:
        if code is None:
            # The homepage currently has no Python quickstart, so there is nothing to check.
            results.append((name, "SKIP", "no data-verify block on the page"))
            continue
        missing = [kind for kind in file_spec.values() if not file_sources.get(kind)]
        if missing:
            results.append((name, "SKIP", f"needs --{missing[0]}"))
            continue
        try:
            inputs = {var: factories[kind]() for var, kind in input_spec.items()}
            files = {fname: file_sources[kind] for fname, kind in file_spec.items()}
            results.append((name, "PASS", run_snippet(code, berg, inputs, files, args.berg_dir)))
        except Exception as exc:  # report and continue with the next snippet
            first = f"{type(exc).__name__}: {exc}".splitlines()[0][:110]
            results.append((name, "FAIL", first))
            if args.traceback:
                traceback.print_exc()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    width = max(len(r[0]) for r in results) if results else 10
    print(f"\n{'snippet':<{width}}  result  detail")
    print("-" * (width + 40))
    for name, status, detail in results:
        print(f"{name:<{width}}  {status:<6}  {detail}")
    counts = {s: sum(r[1] == s for r in results) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")

    if args.write:
        for name, status, _ in results:
            if name in models and status in ("PASS", "FAIL"):
                value = snippet_hash(models[name]["snippet"]) if status == "PASS" else None
                overlay_text = set_verified(overlay_text, name, value)
        OVERLAY.write_text(overlay_text, encoding="utf-8")
        print(f"Updated {OVERLAY.relative_to(REPO)}; now run build_catalog.py")
    if counts["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
