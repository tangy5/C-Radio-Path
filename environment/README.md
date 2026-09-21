# Rebuilding the training environment

Three routes. Route A-container is recommended: a single `docker build`
reproduces the verified environment exactly.

## Route A-container (recommended): Dockerfile in this directory

`environment/Dockerfile` containerizes the verified environment: python 3.10
(`python_version.txt`) + all 180 pins from `pip_freeze_tao_v7_py310.txt`
(`torch==2.11.0+cu126`). The CUDA-12.6 wheels bundle their own CUDA/NCCL
runtime, so the host only needs the NVIDIA driver (run with `--gpus all`).

```bash
docker build -t c-radio-path-exp environment/

# smoke test (1 GPU, 10 steps) inside the container:
docker run --gpus all --rm -it \
    -v /path/to/C-Radio-Path-Exp:/kit -w /kit/run \
    -e NUM_GPUS=1 -e MAX_STEPS=10 \
    c-radio-path-exp ./start_training.sh
```

Note: two dataloader C++ extensions print "not built; using Python fallback"
(FastToTensor, SpatialTransformOps) — the fallbacks were what our runs used;
no action needed. Cropping WSIs additionally needs `openslide-python` + the
OpenSlide system library (data-prep only, not part of the training
container).

## Route A-venv: python 3.10 + pinned wheels

The training environment used for the runs is recorded verbatim in
`pip_freeze_tao_v7_py310.txt` (180 pins, `python_version.txt` = 3.10.20).
Key pins: `torch==2.11.0+cu126`, `pytorch-lightning==2.5.0.post0`,
`hydra-core==1.3.2`, `omegaconf==2.3.0`, `webdataset==0.2.111`,
`timm==1.0.14`, `numpy==2.2.6`, `pandas==2.3.3`.

```bash
python3.10 -m venv tao_env && source tao_env/bin/activate
# torch first, from the CUDA wheel index (keeps the +cu126 build;
# cu126 wheels for cp310 verified available):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
# then everything else from PyPI (torch pin already satisfied):
grep -v "^torch==" pip_freeze_tao_v7_py310.txt | pip install -r /dev/stdin
```

## Route B (reference only — known gaps): upstream TAO docker build

`tao_Dockerfile.reference` is the docker build file from the exported TAO
repo (base `nvcr.io/nvidia/pytorch:25.09-py3`), with `requirements-apt.txt`
and `requirements-pip-*.txt` present under
`../code/tao-pytorch-C-Radio-Path-Exp/docker/`. It documents how the upstream
TAO base image is built, but as shipped it will NOT complete and it does not
target the verified environment:

1. `COPY docker/xformers xformers_src` — the vendored xformers source tree is
   not part of this export; the build stops there. The pinned commit is named
   in the file (`XFORMERS_COMMIT_HASH`) — vendor it yourself or drop the step.
2. Its requirement files target the upstream TAO stack (Python 3 / newer
   CUDA, unpinned), NOT the py3.10 / cu126 verified environment. For exact
   reproduction use Route A.

`v10_release_launch_reference/` holds the original multi-run launch scripts
(paths are site-specific).

## Sanity check after install

From `../run/` (teachers fetched, any small webdataset staged):

```bash
CONFIG_NAME=customer_custom_test NUM_GPUS=1 MAX_STEPS=5 ./start_training.sh
```

(or with the reference spec if the three training webdatasets are staged). A short
loss curve in the log = environment good.
