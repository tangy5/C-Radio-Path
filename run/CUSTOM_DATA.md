# Training on your own webdataset

The distillation pipeline is label-free and format-tolerant. To train C-Radio-Path-Exp
(or any variant) on your own patch collection:

## Shard format contract (minimum)

- One or more `*.tar` shards under a directory (subdirectories are scanned).
- Each tar contains samples grouped by key; each sample needs exactly one
  image member whose extension is one of
  `jpg jpeg png gif webp bmp tiff img` (e.g. `sample_000123.jpg`).
- Extra members (`*.json`, `*.txt`, sha256 sidecars, …) are ignored by the
  training pipeline — captions/labels are NOT required for distillation.
- A shard size of ~10,000 samples is what our runs used (any size works;
  `samples_per_file` in the spec is only used for progress bookkeeping).

Build shards with any webdataset tool, or reuse the bundled generator:
`../data/generation/construct_webdataset.py --src <image_dir> --out <wds_dir>`
(generic directory → shards builder).

## Spec

Copy `experiment_specs/c_radio_path_exp.yaml` to
`experiment_specs/<your_name>.yaml` and edit only the `tar_data_sources`
block (absolute paths are fine — no tokens needed for custom specs):

```yaml
  train_dataset:
    tar_data_sources:
    - root_dir: /absolute/path/to/your_webdataset
      samples_per_file: 10000      # informational
      scale_factor: 1.0            # sampling weight vs other sources
      steps_per_epoch: 2000        # your epoch length (sum over sources)
    student_patch_size: 14
    seed: 42
    ...
```

Multiple sources are mixed by `scale_factor` rates (ours: 1.69 / 0.92 / 10.56
≈ steps 7500 / 4100 / 46900). Keep `steps_per_epoch` ≈
total_samples / global_batch for a natural epoch. Constraint: each source's
`steps_per_epoch` must be ≥ `batch_size` (the trainer's sanity check rejects
otherwise), so for smoke tests use small `dataset.batch_size` AND small
`steps_per_epoch` together.

## Launch

```bash
CONFIG_NAME=<your_name> NUM_GPUS=1 MAX_STEPS=10 ./start_training.sh   # smoke
CONFIG_NAME=<your_name> ./start_training.sh                           # full run
```

The launcher rewrites `__KIT_ROOT__`/`__RESULTS_DIR__` tokens, verifies every
`root_dir` from your spec has shards, checks teachers, and resumes from the
latest checkpoint in the results dir if one exists.

## From scratch vs from a checkpoint

- **From scratch**: leave `resume_training_checkpoint_path: null` — the
  student initializes from the checkpoint named by
  `model.backbone.pretrained_backbone_path` in your spec.
- **Resume**: just rerun the same command — the launcher finds the newest
  non-EMA `latest_epoch_*.pth` in the results dir and patches the spec. To
  resume from a specific checkpoint, set it explicitly in your spec.
- **Continue from an exported backbone** (further training): export format
  ≠ training checkpoint format — set a new init checkpoint or resume only
  from checkpoints produced by this trainer.

Checkpoint cadence for short experiments: `latest_checkpoint_interval`
(default 2000 steps) and `checkpoint_interval` (milestones, default 10
epochs) are the knobs.
