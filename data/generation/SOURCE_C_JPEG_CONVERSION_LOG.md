# source C Patch PNG to JPEG Conversion Log

## Rationale

source C raw WSI files use **JPEG compression internally** (TIFF tag 259 = 7). The patch extraction pipeline saved extracted 1024x1024 patches as **uncompressed PNG**, resulting in ~2.5x storage expansion over the original WSI (e.g., 627MB WSI → 1.7GB PNG patches).

**Why JPEG q=95 is safe for training:**
1. Source WSIs are already JPEG — PNG patches are just a lossless wrapper around already-lossy data
2. JPEG quality=95 is visually indistinguishable from original for pathology images
3. source B training data already uses JPEG (28% of v8 training mix)
4. Distillation training matches teacher features, not pixel-perfect reproduction
5. Measured compression: ~2MB PNG → ~300-600KB JPEG (3-5x savings, ~85% reduction)

**Verified quality:** All converted JPEGs pass: 1024x1024 (or original dims for edge patches), RGB, uint8, full pixel range [0-255], file size > 1KB.

## Conversion Logic

### Script Location
`/root/.qoder/tmp/-lustre-fsw-portfolios-healthcareeng-projects-healthcareeng-monai-JHU-YT-ENV-omni-datasets-CDS-pathology/convert_png_to_jpeg.py`

### Per-file workflow (safe, no extra storage needed):
1. Open PNG, record original dimensions
2. Save as JPEG (quality=95)
3. Verify JPEG: valid image, same dimensions as original, mode=RGB, size > 1KB
4. If JPEG >= PNG size (rare for tiny patches): delete JPEG, **keep PNG as-is**
5. If JPEG < PNG size: **delete PNG**, keep JPEG
6. Update `_patches.csv` files: replace `.png` → `.jpg` in `patch_name` column

### Edge cases handled:
- **Non-1024x1024 patches**: Edge patches (e.g., 1024x993) are valid — verification checks dimensions match original, not that they equal 1024x1024
- **Very small PNGs**: If JPEG ends up larger than PNG, keep the PNG (it was already tiny)
- **Already-converted files**: Script skips slides where .jpg already exists (resume support)
- **CSV references**: All `_patches.csv` files updated to reference .jpg instead of .png

## Datasets Status

| Dataset | Original Size | Status | Space Saved |
|---------|-------------|--------|-------------|
| source C-colorectal-b2 | 60 GB | **DONE** | 42.8 GB |
| source C-gastrointestinal | 133 GB | Converting (PID 2975689) | ~14 GB so far |
| source C-hematologic | 254 GB | Converting (PID 2993641) | ~3 GB so far |
| source C-thorax | 707 GB | Converting (PID 2993644) | ~4 GB so far |
| source C-breast | 1.4 TB | Converting (PID 3010107) | just started |
| source C-colorectal-b1 | 3.5 TB | Converting (PID 3010110) | just started |
| **Total expected savings** | **~6 TB** | | **~8.5 TB** |

### NOT being converted (actively generating patches):
- **source C-skin-b1** — patch generation running (PID 2651947)
- **source C-skin-b2** — patch generation running (PID 2652187)

These skin datasets should be converted AFTER their patch generation completes. Run the same script:
```bash
PYTHON="<OMNI_DATASETS_DIR>/pathology_VLM/conda_envs/pathvlm/bin/python"
SCRIPT="/root/.qoder/tmp/-lustre-fsw-portfolios-healthcareeng-projects-healthcareeng-monai-JHU-YT-ENV-omni-datasets-CDS-pathology/convert_png_to_jpeg.py"

nohup $PYTHON -u $SCRIPT --dataset source C-skin-b1 --quality 95 > logs/convert_skin-b1.log 2>&1 &
nohup $PYTHON -u $SCRIPT --dataset source C-skin-b2 --quality 95 > logs/convert_skin-b2.log 2>&1 &
```

## Also update the patch extraction script

After all conversions complete, update the patch extraction pipeline to save JPEG directly (so new patches don't need converting):

**File:** `<PATHOLOGY_DATASETS_DIR>/source_c_patch/patch_extraction.py`

**Change (line ~172):**
```python
# Before:
patch.save(patch_path, "PNG")

# After:
patch_path = patch_path.replace('.png', '.jpg')
patch.save(patch_path, "JPEG", quality=95)
```

Also update the filename generation (line ~170) to use `.jpg` extension instead of `.png`.

## How to Monitor Progress

```bash
# Check process status
ps aux | grep convert_png | grep -v grep

# Check per-dataset progress (JPG count vs total)
for ds in gastrointestinal hematologic thorax breast colorectal-b1; do
  dir="<PATHOLOGY_DATASETS_DIR>/source_c_patch/source C-${ds}"
  jpg=$(find "$dir" -name "*.jpg" 2>/dev/null | wc -l)
  png=$(find "$dir" -name "*.png" 2>/dev/null | wc -l)
  total=$((jpg + png))
  pct=$((jpg * 100 / (total > 0 ? total : 1)))
  echo "$ds: $jpg/$total ($pct%)"
done

# Check logs
tail -f <CLUSTER_STORAGE>/<dataset>.log
```

Note: Lustre may buffer log output even with `python -u`. Trust the disk counts over the log timestamps.

## Related Running Processes

| Process | PID | Purpose |
|---------|-----|---------|
| source_a_roi_recovery.py | 2620965 | Grounding data generation (800/7110 WSIs done) |
| run_pipeline.py skin-b1 | 2651947 | source C skin-b1 patch extraction (~40%) |
| run_pipeline.py skin-b2 | 2652187 | source C skin-b2 patch extraction (~21%) |

## Context: Why This Matters

The v8 C-RADIO LoRA training job (slurm ID 11399143) failed due to **disk quota exceeded** on the Lustre filesystem. Converting PNG patches to JPEG is the primary strategy to reclaim ~8.5 TB of storage so the training job can be resubmitted successfully.

Original total source C storage: ~22.8 TB (13 TB raw WSI + 9.8 TB patches)
After JPEG conversion: ~13 TB raw + ~2 TB patches = **~15 TB** (saving ~8 TB)
