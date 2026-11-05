#!/usr/bin/env python3
"""
Generate the homepage overview figure: example images (body, scene, face and optionally
object) and the fMRI responses BERG predicts for them in early visual and category-selective
areas. Each category-selective area is only shown when its category's image is given.

Uses fmri-things_fmri_1-vit_b_32 for all three THINGS fMRI1 subjects. For each
subject, one model is loaded with all areas selected; its output is split back into
areas using the ROI sizes from the model metadata, and averaged over each area's voxels
(left and right hemisphere pooled). The training data were z-scored per voxel within each
scanning session, so 0 is a voxel's average response to the THINGS images.

Output
------
docs/index.html: the predicted values and image paths, embedded between
<!-- overview:data:start/end --> and drawn by docs/assets/js/overview.js.

The images must be inside docs/ (e.g. docs/img/) and ones we may publish. The page shows
them as they are, cropped to a square on screen; the model receives the same centre crop.
For a photo whose licence forbids changes (e.g. CC BY-ND), pass --uncropped <category>
so the page shows it whole, letterboxed.

Usage
-----
    python scripts/website/make_overview_figure.py \\
        --berg-dir /path/to/brain-encoding-response-generator \\
        --body docs/img/body.jpg --scene docs/img/scene.jpg --face docs/img/face.jpg \\
        --body-alt "Torso and arms of a person in a T-shirt" ...

Check the printed table before publishing, and update the figure caption in
docs/index.html if the pattern changes.

Note: THINGS fMRI1 shows only object photos and its responses are z-scored, so a typical
object image predicts roughly average (near 0) responses everywhere, LOC included.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
INDEX_HTML = DOCS / "index.html"
MODEL_ID = "fmri-things_fmri_1-vit_b_32"

# key, model ROIs pooled into the area, full name, image category the area prefers
AREAS = [
    ("V1", ["V1"], "Primary visual cortex", None),
    ("V2", ["V2"], "Visual area 2", None),
    ("V3", ["V3"], "Visual area 3", None),
    ("hV4", ["hV4"], "Human visual area 4", None),
    ("LOC", ["lLOC", "rLOC"], "Lateral occipital complex", "object"),
    ("EBA", ["lEBA", "rEBA"], "Extrastriate body area", "body"),
    ("PPA", ["lPPA", "rPPA"], "Parahippocampal place area", "scene"),
    ("FFA", ["lFFA", "rFFA"], "Fusiform face area", "face"),
]
CATEGORIES = [("object", "Object"), ("body", "Body"), ("scene", "Scene"), ("face", "Face")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--berg-dir", required=True, help="local copy of brain-encoding-response-generator")
    for key, label in CATEGORIES:
        p.add_argument(f"--{key}", help=f"{label.lower()} image, inside docs/")
        p.add_argument(f"--{key}-alt", help=f"alt text for the {label.lower()} image")
    p.add_argument("--uncropped", nargs="+", default=[], choices=[key for key, _ in CATEGORIES],
                   help="show these images whole instead of square-cropped (for no-derivatives licences)")
    p.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--device", default="auto")
    p.add_argument("--model-size", type=int, default=224, help="side length of the image passed to the model")
    p.add_argument("--no-page", action="store_true", help="don't write into docs/index.html")
    p.add_argument("--json-out", help="also write the figure data to this file")
    args = p.parse_args()
    args.categories = [(key, label) for key, label in CATEGORIES if getattr(args, key)]
    if len(args.categories) < 2:
        p.error("give at least two of --object, --body, --scene, --face")
    for key, _ in args.categories:
        path = Path(getattr(args, key)).resolve()
        if DOCS not in path.parents:
            p.error(f"--{key}: the image must be inside docs/ so the page can show it ({path})")
        setattr(args, key, path)
    keys = {key for key, _ in args.categories}
    args.areas = [area for area in AREAS if area[3] is None or area[3] in keys]
    return args


def square_crop(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


def to_model_input(crops, size):
    """[n, 3, size, size] uint8, as the model card asks for."""
    arrays = [np.asarray(c.resize((size, size), Image.LANCZOS), dtype=np.uint8).transpose(2, 0, 1) for c in crops]
    return np.stack(arrays)


def predict_area_means(berg, images_uint8, subjects, device, areas=AREAS):
    """Mean predicted response per area. Returns values [n_subjects, n_images, n_areas] and voxel counts."""
    rois = [roi for _, area_rois, _, _ in areas for roi in area_rois]
    values = np.full((len(subjects), len(images_uint8), len(areas)), np.nan)
    voxels = np.zeros((len(subjects), len(areas)), dtype=int)
    for s, subject in enumerate(subjects):
        metadata = berg.get_model_metadata(MODEL_ID, subject=subject)
        sizes = [len(metadata["roi"][roi]) for roi in rois]
        model = berg.get_encoding_model(MODEL_ID, subject=subject, selection={"roi": rois}, device=device)
        responses = np.asarray(berg.encode(model, images_uint8, show_progress=False))   # [n_images, n_voxels]
        if responses.shape[1] != sum(sizes):
            raise RuntimeError(f"subject {subject}: {responses.shape[1]} voxels returned, ROI sizes add up to {sum(sizes)}")
        per_roi = dict(zip(rois, np.split(responses, np.cumsum(sizes)[:-1], axis=1)))
        for a, (_, area_rois, _, _) in enumerate(areas):
            pooled = np.concatenate([per_roi[roi] for roi in area_rois], axis=1)
            voxels[s, a] = pooled.shape[1]
            if pooled.shape[1]:
                values[s, :, a] = pooled.mean(axis=1)
        if hasattr(model, "cleanup"):
            model.cleanup()
        del model, responses, per_roi
        gc.collect()
    return values, voxels


def figure_data(args, values, voxels):
    mean = np.nanmean(values, axis=0)            # [n_images, n_areas]
    r = lambda x: None if np.isnan(x) else round(float(x), 3)
    return {
        "model": MODEL_ID,
        "subjects": args.subjects,
        "units": "In silico response (z)",
        "areas": [
            {"key": key, "name": name, "category": category, "voxels": voxels[:, a].tolist()}
            for a, (key, _, name, category) in enumerate(args.areas)
        ],
        "images": [
            {
                "key": key,
                "label": label,
                "src": getattr(args, key).relative_to(DOCS).as_posix(),
                "alt": getattr(args, f"{key}_alt") or f"Example {label.lower()} image",
                "fit": "contain" if key in args.uncropped else "cover",
                "values": [r(v) for v in mean[i]],
                "subjects": [[r(v) for v in values[s, i]] for s in range(len(args.subjects))],
            }
            for i, (key, label) in enumerate(args.categories)
        ],
    }


def print_table(data):
    keys = [a["key"] for a in data["areas"]]
    print(f"{MODEL_ID}, subjects {data['subjects']}, mean over voxels and subjects (z)\n")
    print(f"{'image':<8}" + "".join(f"{k:>8}" for k in keys) + "   highest category area")
    category_areas = [i for i, a in enumerate(data["areas"]) if a["category"]]
    for image in data["images"]:
        row = image["values"]
        best = max(category_areas, key=lambda i: -np.inf if row[i] is None else row[i])
        expected = data["areas"][best]["category"] == image["key"]
        print(f"{image['label']:<8}" + "".join(f"{'-' if v is None else f'{v:.2f}':>8}" for v in row)
              + f"   {keys[best]}{'' if expected else '  (not the expected area)'}")
    print("\nvoxels per subject: " + ", ".join(f"{a['key']} {a['voxels']}" for a in data["areas"]))


def embed_in_page(data, page_path: Path):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    block = f'          <script type="application/json" id="overview-data">{payload}</script>'
    page = page_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- overview:data:start -->)(.*?)(\n[ \t]*<!-- overview:data:end -->)", re.S)
    if not pattern.search(page):
        sys.exit(f"{page_path}: markers overview:data:start/end not found")
    page_path.write_text(pattern.sub(lambda m: m.group(1) + "\n" + block + m.group(3), page, count=1), encoding="utf-8")


def main():
    args = parse_args()
    from berg import BERG

    crops = [square_crop(getattr(args, key)) for key, _ in args.categories]
    berg = BERG(args.berg_dir)
    values, voxels = predict_area_means(berg, to_model_input(crops, args.model_size), args.subjects, args.device, args.areas)
    data = figure_data(args, values, voxels)
    print_table(data)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if not args.no_page:
        embed_in_page(data, INDEX_HTML)
        print(f"\nWrote the data into {INDEX_HTML.relative_to(REPO)}")
    if any(getattr(args, f"{key}_alt") is None for key, _ in args.categories):
        print("Some images have no alt text yet; pass --body-alt etc. or edit the data in docs/index.html.")


if __name__ == "__main__":
    main()
