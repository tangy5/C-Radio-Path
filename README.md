# C-Radio-Path-Exp

> pathology foundation model for embedding

The C-Radio-Path training kit: TAO training code, a reference experiment
spec, an optional init checkpoint (H-optimus-0), a downloader for
example-teacher weights, and manifests + builder chains for the ~13.17M-image
pretraining webdataset mix. Export a trained milestone with
`export/export_backbone.py`.

## What's included

| Component | Notes |
|---|---|
| Training code (`code/tao-pytorch-C-Radio-Path-Exp/`) | NVIDIA TAO + pathology extensions, Apache-2.0 |
| Entrypoint + launcher (`run/`) | `distill_fastval_ddp.py`, `train.py`, `start_training.sh` |
| Reference spec (`run/experiment_specs/c_radio_path_exp.yaml`) | example configuration — edit to your needs |
| Environment | `environment/Dockerfile` (one build) or the venv route in `environment/README.md` |

Not bundled: teacher weights (`teachers/` fetch scripts) and the pretraining
webdatasets (manifests + builder chains in `data/`). Datasets and teacher
models are third-party assets: do your own data-use and license review.

## Quickstart

```bash
# prerequisites: GPUs (8x for the recipe; 1 for smoke), webdatasets staged
# (data/MANIFEST.md), teacher weights fetched, env built (below)
cd run
NUM_GPUS=1 MAX_STEPS=10 ./start_training.sh   # smoke test (1 GPU, 10 steps)
./start_training.sh                            # full training (8x GPU, resumable)
# train on your own webdataset: see CUSTOM_DATA.md
```

Data is found via `DATASET_ROOT` (default `<kit>/data/webdatasets`); results
go to `RESULTS_DIR` (default `run/results/c_radio_path_exp`).

## Environment (Docker)

```bash
docker build -t c-radio-path-exp environment/

# smoke test inside the container:
docker run --gpus all --rm -it \
    -v "$PWD":/kit -w /kit/run \
    -e NUM_GPUS=1 -e MAX_STEPS=10 \
    c-radio-path-exp ./start_training.sh
```

The pinned `torch==2.11.0+cu126` wheels bundle their CUDA/NCCL runtime — the
host only needs the NVIDIA driver. Venv route and upstream TAO Dockerfile
reference: `environment/README.md`.

## Reference training configuration

- Student: `vit_giant_patch14_reg4_dinov2` (ViT-G/14, 1,135M params), full
  fine-tune; optional init from the bundled checkpoint via the spec.
- Example teachers: Prov-GigaPath λ=0.6 (ImageNet norm) + H-optimus-0 λ=0.4
  (H&E norm), frozen, input 224 — substitute your own.
- Loss: combo (cosine summary + dampened-MSE spatial), projection MLP 2048×2.
- AdamW lr 3e-5, cosine, warmup 2 ep, wd 0.05, grad-clip 1.0, fp16, EMA
  0.99998, seed 42; 100 epochs × 58,500 steps, global batch 256.
- Checkpoints: milestone every 10 epochs + `latest` every 2,000 steps.

## Teacher models & licensing

Teacher entries in the specs are **examples**, not statements about any
training run; the framework is teacher-agnostic. Every teacher is a
third-party model governed by its own license — you are responsible for
obtaining weights and ensuring your use (including distilled embeddings)
complies. No third-party weights are redistributed here.

| Example teacher | Weights |
|---|---|
| Prov-GigaPath | `teachers/fetch_gigapath.sh` (places weights in `teachers/GigaPath/`) |
| H-optimus-0 | HuggingFace `bioptimus/H-optimus-0` — place `pytorch_model.bin` in `teachers/H-optimus-0/`; sha256 in `teachers/checksums.txt` |

## Layout

```
C-Radio-Path-Exp/
├── code/tao-pytorch-C-Radio-Path-Exp/  # TAO training codebase (Apache-2.0)
├── run/
│   ├── start_training.sh               # launcher (relocatable, resume-aware)
│   ├── CUSTOM_DATA.md                  # training on your own webdataset
│   ├── experiment_specs/c_radio_path_exp.yaml
│   └── scripts/                        # distill_fastval_ddp.py + pipeline
├── export/export_backbone.py           # milestone -> backbone export
├── teachers/                           # fetch script + checksums
├── data/                               # webdataset manifests + builder chains
└── environment/                        # Dockerfile (verified env) + references
```
