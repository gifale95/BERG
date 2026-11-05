#!/usr/bin/env python3
"""
Build the website model catalog from BERG's model cards.

Sources
-------
- berg/models/model_cards/*.yaml   facts: id, modality, dataset, species, stimuli, architecture
- scripts/website/models_overlay.json   titles, short labels, groups, vocabulary, snippets
- docs/models/overview.rst   the 🏆 (formerly ✅) "best model per dataset" marker, used for ordering only

Outputs
-------
- the static model directory and an embedded copy of the catalog inside
  docs/index.html, between <!-- catalog:list|json:start/end --> markers,
  so the page shows every model without JavaScript.

The script stops with a list of problems if the cards, the overlay, the vocabulary
or the docs disagree. A snippet is only exported when snippet_verified holds the hash
that verify_snippets.py recorded for its current text.

Usage
-----
    python scripts/website/build_catalog.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CARDS_DIR = REPO / "berg" / "models" / "model_cards"
OVERLAY = REPO / "scripts" / "website" / "models_overlay.json"
OVERVIEW_RST = REPO / "docs" / "models" / "overview.rst"
DOCS_CARDS_DIR = REPO / "docs" / "models" / "model_cards"
INDEX_HTML = REPO / "docs" / "index.html"

REQUIRED_OVERLAY_FIELDS = ("title", "dataset", "architecture", "group", "order")
CONTROL_NOTE = "Untrained control"
BRAINSCORE_NOTE = "Weights hosted by BrainScore"


def load_cards(cards_dir: Path, problems: list[str]) -> dict[str, dict]:
    cards = {}
    for path in sorted(cards_dir.glob("*.yaml")):
        if path.stem == "template":
            continue
        with open(path, encoding="utf-8") as f:
            card = yaml.safe_load(f)
        model_id = str(card.get("model_id", "")).strip()
        if model_id != path.stem:
            problems.append(f"{path.name}: model_id '{model_id}' does not match the file name")
        cards[path.stem] = card
    return cards


def load_preferred(overview_rst: Path) -> set[str]:
    text = overview_rst.read_text(encoding="utf-8")
    return set(re.findall(r"\*\s+-\s+(?:🏆|✅)\s*\n\s+-\s+:doc:`model_cards/([^`]+)`", text))


def alias_lookup(entries: list[dict]) -> dict[str, dict]:
    lookup = {}
    for entry in entries:
        for alias in entry.get("aliases", []):
            lookup[alias] = entry
    return lookup


def snippet_hash(snippet: str) -> str:
    """Must match verify_snippets.snippet_hash."""
    return hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:12]


def build_models(cards, overlay, preferred, problems, warnings):
    vocab = overlay["vocabulary"]
    modality_by_alias = alias_lookup(vocab["modality"])
    species_by_alias = alias_lookup(vocab["species"])
    stimulus_keys = {s["key"] for s in vocab["stimulus_types"]}
    stimuli_aliases = vocab["stimuli_aliases"]
    group_rank = {g["key"]: i for i, g in enumerate(overlay["groups"])}
    entries = overlay["models"]

    for model_id in sorted(set(cards) - set(entries)):
        problems.append(f"{model_id}: model card has no overlay entry")
    for model_id in sorted(set(entries) - set(cards)):
        problems.append(f"{model_id}: overlay entry has no model card")
    for model_id in sorted(preferred - set(cards)):
        problems.append(f"{model_id}: marked 🏆 in overview.rst but has no model card")

    models = []
    for model_id in sorted(set(cards) & set(entries)):
        card, extra = cards[model_id], entries[model_id]
        missing = [k for k in REQUIRED_OVERLAY_FIELDS if k not in extra]
        if missing:
            problems.append(f"{model_id}: overlay is missing {', '.join(missing)}")
            continue

        raw_modality = str(card.get("modality", "")).strip()
        raw_species = str(card.get("species", "")).strip()
        raw_stimuli = str(card.get("stimuli", "")).strip()
        modality = modality_by_alias.get(raw_modality)
        species = species_by_alias.get(raw_species)
        stimuli = stimuli_aliases.get(raw_stimuli)
        if modality is None:
            problems.append(f"{model_id}: modality '{raw_modality}' is not in the vocabulary")
        if species is None:
            problems.append(f"{model_id}: species '{raw_species}' is not in the vocabulary")
        if stimuli is None:
            problems.append(f"{model_id}: stimuli '{raw_stimuli}' has no entry in stimuli_aliases")
        elif not set(stimuli["types"]) <= stimulus_keys:
            problems.append(f"{model_id}: stimuli_aliases uses unknown types {stimuli['types']}")
        if extra["group"] not in group_rank:
            problems.append(f"{model_id}: group '{extra['group']}' is not defined in groups")
        if not (DOCS_CARDS_DIR / f"{model_id}.rst").exists():
            problems.append(f"{model_id}: no docs page docs/models/model_cards/{model_id}.rst")
        if None in (modality, species, stimuli) or extra["group"] not in group_rank:
            continue

        provenance = extra.get("provenance", "berg")
        if provenance not in ("berg", "brainscore"):
            problems.append(f"{model_id}: provenance must be 'berg' or 'brainscore'")
        snippet = extra.get("snippet")
        mark = extra.get("snippet_verified")
        verified = bool(snippet) and isinstance(mark, str) and mark == snippet_hash(snippet)
        if snippet and isinstance(mark, str) and not verified:
            warnings.append(f"{model_id}: snippet changed since it was verified; not published until re-verified")
        elif mark is True:
            warnings.append(f"{model_id}: snippet_verified must be set by verify_snippets.py; not published")

        models.append({
            "id": model_id,
            "title": extra["title"],
            "dataset": extra["dataset"],
            "architecture": extra["architecture"],
            "variant_note": extra.get("variant_note"),
            "modality": modality["key"],
            "species": species["key"],
            "stimulus_types": stimuli["types"],
            "stimulus_label": stimuli["label"],
            "group": extra["group"],
            "provenance": provenance,
            "provenance_note": BRAINSCORE_NOTE if provenance == "brainscore" else None,
            "docs_url": overlay["docs_base"] + f"{model_id}.html",
            "snippet": snippet if verified else None,
            "snippet_verified": verified,
            "dataset_full": str(card.get("training_dataset", "")).strip(),
            "architecture_full": str(card.get("model_type", card.get("model_architecture", ""))).strip(),
            "_sort": (
                group_rank[extra["group"]],
                extra.get("variant_note") == CONTROL_NOTE,
                model_id not in preferred,
                extra["order"],
            ),
            "_preferred": model_id in preferred,
        })

    models.sort(key=lambda m: m["_sort"])
    return models


# ---------------------------------------------------------------- static HTML

def esc(text) -> str:
    return html.escape(str(text), quote=True)


def label_maps(overlay):
    vocab = overlay["vocabulary"]
    return (
        {m["key"]: m["label"] for m in vocab["modality"]},
        {s["key"]: s["label"] for s in vocab["species"]},
            {s["key"]: s["label"] for s in vocab["stimulus_types"]},
    )


# Directory sections: BERG's native models, then the BrainScore integration.
# Wording follows the README and the documentation.
SECTIONS = [
    ("berg", "BERG's native models",
     "Encoding models that are part of BERG, each documented with a model card in the documentation."),
    ("brainscore", "BrainScore models",
     "Beyond BERG's native models, BERG is also integrated with BrainScore, giving you access to hundreds of "
     "vision models scored against macaque neural recordings (V1, V2, V4, IT), as well as GPT-family language "
     "models scored against human fMRI data. These models are hosted by BrainScore."),
]


def render_row(m, labels, indent, group_start):
    modality_label, species_label, _ = labels
    classes = "model group-start" if group_start else "model"
    note = f'<span class="model-note">{esc(m["variant_note"])}</span>' if m["variant_note"] else ""
    lines = [
        f'<tr class="{classes}" id="model-{esc(m["id"])}">',
        '  <th scope="row" class="model-name">',
        f'    <span class="model-dataset">{esc(m["dataset"])}</span>',
        f'    <span class="model-arch">{esc(m["architecture"])}{note}</span>',
        f'    <code class="model-id">{esc(m["id"])}</code>',
        '  </th>',
        f'  <td data-label="Modality">{esc(modality_label[m["modality"]])}</td>',
        f'  <td data-label="Species">{esc(species_label[m["species"]])}</td>',
        f'  <td data-label="Stimuli">{esc(m["stimulus_label"])}</td>',
        f'  <td class="model-links"><a href="{esc(m["docs_url"])}">Model card<span class="visually-hidden"> for {esc(m["id"])}</span></a></td>',
        '</tr>',
    ]
    return "\n".join(indent + line for line in lines)


def render_list(models, overlay, indent="            "):
    labels = label_maps(overlay)
    out = []
    for key, heading, intro in SECTIONS:
        section = [m for m in models if m["provenance"] == key]
        if not section:
            continue
        rows, previous = [], None
        for m in section:
            rows.append(render_row(m, labels, indent + "      ", previous is not None and m["group"] != previous))
            previous = m["group"]
        out += [
            f'{indent}<div class="directory-section" data-provenance="{key}">',
            f'{indent}  <h3>{esc(heading)}</h3>',
            f'{indent}  <p class="directory-intro">{esc(intro)}</p>',
            f'{indent}  <table class="model-table">',
            f'{indent}    <thead>',
            f'{indent}      <tr><th scope="col">Training dataset and model</th><th scope="col">Recording modality</th>'
            f'<th scope="col">Species</th><th scope="col">Stimuli</th>'
            f'<th scope="col"><span class="visually-hidden">Documentation</span></th></tr>',
            f'{indent}    </thead>',
            f'{indent}    <tbody>',
            *rows,
            f'{indent}    </tbody>',
            f'{indent}  </table>',
            f'{indent}</div>',
        ]
    return "\n".join(out)


def render_inline_json(catalog, indent="          "):
    """The catalog embedded in the page, so filters also work when index.html is opened from disk."""
    data = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'{indent}<script type="application/json" id="catalog-data">{data}</script>'


def replace_between(text, name, content, problems):
    pattern = re.compile(
        rf"(<!-- catalog:{name}:start -->)(.*?)(\n[ \t]*<!-- catalog:{name}:end -->)", re.S
    )
    if not pattern.search(text):
        problems.append(f"docs/index.html: markers catalog:{name}:start/end not found")
        return text
    return pattern.sub(lambda m: m.group(1) + "\n" + content + m.group(3), text, count=1)


# ---------------------------------------------------------------------- main

def print_summary(models, overlay):
    modality_label, species_label, _ = label_maps(overlay)
    header = f"{'model id':<50} {'modality':<27} {'species':<8} {'stimuli':<20} {'🏆':<2} snippet"
    print(header)
    print("-" * len(header))
    for m in models:
        print(
            f"{m['id']:<50} {modality_label[m['modality']]:<27} {species_label[m['species']]:<8} "
            f"{','.join(m['stimulus_types']):<20} {'🏆' if m['_preferred'] else '  ':<2} "
            f"{'verified' if m['snippet_verified'] else '-'}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = parser.parse_args()

    problems: list[str] = []
    warnings: list[str] = []
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    cards = load_cards(CARDS_DIR, problems)
    preferred = load_preferred(OVERVIEW_RST)
    if not preferred:
        problems.append("docs/models/overview.rst: no 🏆 markers found (format changed?)")
    models = build_models(cards, overlay, preferred, problems, warnings)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    vocab = overlay["vocabulary"]
    catalog = {
        "source": "berg/models/model_cards + scripts/website/models_overlay.json",
        "vocabulary": {
            "modality": [{"key": e["key"], "label": e["label"]} for e in vocab["modality"]],
            "species": [{"key": e["key"], "label": e["label"]} for e in vocab["species"]],
            "stimulus_types": [{"key": e["key"], "label": e["label"]} for e in vocab["stimulus_types"]],
        },
        "groups": overlay["groups"],
        "models": [{k: v for k, v in m.items() if not k.startswith("_")} for m in models],
    }
    page = INDEX_HTML.read_text(encoding="utf-8")
    page = replace_between(page, "json", render_inline_json(catalog), problems)
    page = replace_between(page, "list", render_list(models, overlay), problems)

    if problems:
        print(f"Catalog not written: {len(problems)} problem(s)\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print_summary(models, overlay)
    n_verified = sum(m["snippet_verified"] for m in models)
    print(f"\n{len(models)} models, {len(overlay['groups'])} groups, {n_verified} verified snippets")
    if args.check:
        print("Check passed (nothing written).")
        return

    INDEX_HTML.write_text(page, encoding="utf-8")
    print(f"Refreshed the static list and embedded catalog in {INDEX_HTML.relative_to(REPO)}")


if __name__ == "__main__":
    main()
