# Tests

The suite is split into three tiers. The first two need no model weights and
run on every PR; the third needs the real weights and runs manually.

| Tier | Marker | Needs weights? | What it checks |
|------|--------|----------------|----------------|
| Smoke / unit | *(none)* | no | import & registration, catalog/list/describe, YAML cards, validators, param errors, `generate_response` with stubbed weights |
| Feature extraction | `heavy` | no (downloads torchvision backbone) | the real ViT/AlexNet backbones build and output the right shapes |
| Integration | `weights` | yes | real `encode()`, plus selection/ROI outputs match a slice of the full output |

By default `pytest` runs everything except the `weights` tier (set in
`pyproject.toml`).

## Running

```bash
pytest                          # everything except weights (the usual run)
pytest -m "not heavy"           # skip the backbone download too
pytest -m heavy                 # only the backbone tests
```

`-m heavy` selects *only* tests tagged `heavy`; the rest show up as
"deselected", which just means filtered out — not failed.

## Running the full suite (with weights)

The weights tier is off by default and also skips unless `BERG_DIR` points at a
real weight download. To run it, point `BERG_DIR` at your local copy:

```bash
# weights tier only
BERG_DIR=/path/to/brain-encoding-response-generator pytest -m weights

# or the whole thing at once (clears the default -m filter)
BERG_DIR=/path/to/brain-encoding-response-generator pytest -o addopts=""
```

A weights test skips per model if that model's files aren't present, so a
partial download works — only the models you have are tested. Don't have the
data? Grab one model from the public bucket (no AWS account needed):

```bash
DEST=berg_dir/encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32
aws s3 cp --no-sign-request --recursive --exclude "*" --include "*subject-01.npy" \
  s3://brain-encoding-response-generator/encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32 \
  "$DEST"
BERG_DIR=$PWD/berg_dir pytest -m weights
```
