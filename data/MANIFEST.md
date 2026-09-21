# Pretraining webdatasets — manifest & staging guide

C-Radio-Path-Exp trains on the "v8 mix": three webdatasets, ~13.17M images,
1,317 shards, ~7.6 TB. The shards are NOT bundled with this kit (size);
stage them separately, keep the directory names, and verify with
`verify_webdatasets.py`.

| Dataset | Shards | Bytes | Samples | Member format |
|---|---|---|---|---|
| `source_a_webdataset/` | 169 (in `1000/`) | 1.66 TB | ~1.69M | `<key>.png` + `<key>.txt` + `<key>.json{sha256, caption}` |
| `source_b_webdataset/` | 92 (`1000/`56, `2000/`31, `3000/`4, `4000/`1) | 0.04 TB | ~920K | `<key>.jpg` + `<key>.json{sha256}` |
| `source_c_webdataset/` | 1,056 (`1000/`1000, `2000/`56) | 5.90 TB | 10,556,849 (exact) | `<key>.jpg` + `<key>.json{sha256}` |

All shards: 10,000 samples each, written by the generators in `generation/`
(seed 42). The per-dataset file lists with byte sizes are in
`manifests/{source_a,source_b,source_c}.tsv` (`size<TAB>relative/path`).

## Staging + verification

```bash
# stage the three directories (names must match exactly) under a root, then:
python verify_webdatasets.py --root /path/to/staging_root          # sizes+presence
python verify_webdatasets.py --root /path/to/staging_root --open 5 # open random shards
```

`DATASET_ROOT` for `run/start_training.sh` must point at that staging root
(the parent containing the three directories).

## Provenance

- **source A** — patch + caption pairs from the source A-1.6M pipeline;
  built by `generation/construct_webdataset_resumable.py` (+ resumable
  runner) from `source_a_output`; guide: `WEBDATASET_CONVERSION_GUIDE.md`.
  The original acquisition pipeline that produced `source_a_output` is
  not part of this workspace — the shards (manifest above) or the source
  directory are the transferable artifacts; document external provenance
  before any re-generation.
- **source B** — public source B-1M images (resized copy under
  `source_b1M/source_b1M_resize`); sharded by `generation/construct_webdataset.py`
  (generic directory → webdataset builder; it does NOT need the source B lookup
  table). The 2.2 GB `source_b_1M_lookup.csv` (source B-1M metadata) is NOT bundled;
  transfer separately only if you need per-image source B metadata.
- **source C** — full chain included in `generation/source_c_patch/` (16 scripts):
  source C-500 WSIs → case selection → 1024×1024 patch extraction →
  foreground/QC filtering → JPEG q95 conversion (rationale in our
  `SOURCE_C_JPEG_CONVERSION_LOG.md`) → sharding by
  `construct_source_c_webdataset.py` (8 organ subsets; exact final count
  10,556,849) → `validate_source_c_webdataset.py` post-check.

## Data-use review required before transfer

source A is derived from a controlled-access source (data-use policies apply); source B and source C
carry their own upstream licenses/terms. Confirm redistribution rights for
the customer's jurisdiction and use case before shipping any shard bytes —
this kit intentionally ships only manifests and generators.
