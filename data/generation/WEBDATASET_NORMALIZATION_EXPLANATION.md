# WebDataset Normalization Changes: What the TAO Statement Means

## The Statement
> "If the teacher/student input normalization is correctly applied (we made some changes to make sure this can be handled more consistently, as we made some changes to the webdataset loader in the earlier commits)."

## What This Means

### Background: Two Different DataLoader Behaviors

TAO has two dataloader paths with DIFFERENT normalization behavior:

#### 1. Regular CLDataset (Non-WebDataset)
```python
# Regular image folder dataset
images_dir: /path/to/images

# In augmentation.py:
class CLDataAugmentation:
    def __call__(self, imgs, ...):
        # ... augmentation ...
        imgs = [TF.to_tensor(img) for img in imgs]  # [0,255] -> [0,1]
        imgs = [TF.normalize(img, mean=self.mean, std=self.std)  # Apply normalization
                 for img in imgs]
        return imgs
```
**Output**: Images are ALREADY normalized (e.g., H-optimus: mean=[0.707, 0.579, 0.704])

#### 2. WebDataset (Equivariant Pipeline) - THE NEW WAY
```python
# WebDataset config
tar_data_sources:
  - root_dir: /path/to/shards
    
# In data_pipeline.py:
def fast_to_tensor(pic: Image.Image) -> torch.Tensor:
    """Convert PIL Image to CHW float32 tensor in [0, 1]"""
    np_img = np.array(pic, copy=False)
    img = torch.from_numpy(np_img).permute(2, 0, 1)
    fp_img.div_(255)  # ONLY scales to [0,1], NO normalization applied!
    return fp_img
```
**Output**: Images are in **[0, 1] range, NOT normalized**

## The Problem (Before)

### Inconsistent Normalization Paths

| Path | Output Range | Normalization |
|------|--------------|---------------|
| CLDataset | Normalized tensor | Applied in augmentation.py |
| Old WebDataset | Inconsistent | Sometimes applied, sometimes not |

This caused issues:
- Teachers expected different normalizations
- Student used dataset's normalization
- Hard to align teacher/student inputs

## The Solution (After - Current Code)

### Unified Approach: Always Output [0, 1]

**WebDataset now consistently outputs raw [0, 1] images:**

```python
# In data_pipeline.py (WebDataset)
img = fast_to_tensor(sample)  # Only converts to [0, 1], no normalization
```

**Per-teacher normalization applied in distiller:**

```python
# In distiller.py
class ClassDistiller:
    def _apply_teacher_normalization(self, teacher_input, teacher_config, device):
        """Normalize teacher input from [0,1] to per-teacher normalization."""
        # All inputs are now guaranteed to be in [0, 1]
        mean_t = torch.tensor(teacher_config['norm_mean'], ...)
        std_t = torch.tensor(teacher_config['norm_std'], ...)
        return (teacher_input - mean_t) / std_t
```

## What Changed in WebDataset Loader

### Before (Inconsistent)
```python
# Old code might have:
- Applied dataset-level normalization in pipeline
- Or not, depending on config
- Hard to track what was normalized
```

### After (Consistent)
```python
# New code guarantees:
- WebDataset ALWAYS outputs [0, 1] images
- Normalization is ALWAYS applied per-teacher in distiller
- Student gets [0, 1] and applies its own normalization
```

## How It Works Now

### Data Flow

```
WebDataset (tar files)
    ↓
[Load image bytes]
    ↓
[fast_to_tensor] → Output: [0, 1] range tensor
    ↓
[Apply augmentations] → Still [0, 1] range
    ↓
[Batch] → batch["img"] = [0, 1] tensors
    ↓
Distiller Training Step:
    ├── Student path:
    │   └── student_summary, student_spatial = model(batch["img"], return_features=True)
    │       └── Inside model: Applies student normalization (H-optimus: [0.707, 0.579, 0.704])
    │
    └── Teacher paths (for each teacher):
        ├── teacher_input = _apply_teacher_normalization(batch["img"], teacher_config)
        │   └── H-optimus-0: [0.707, 0.579, 0.704]
        │   └── Prov-GigaPath: [0.485, 0.456, 0.406]
        │   └── UNI2-h: [0.485, 0.456, 0.406]
        │   └── Prov-GigaPath: [0.485, 0.456, 0.406]
        └── loss = loss_fn(..., teacher_batch_input=teacher_input)
```

## Why This Is Better

### 1. Consistency
- WebDataset and regular dataset can use same normalization logic
- No more "is this normalized?" confusion

### 2. Flexibility
- Each teacher can have its own normalization
- Student can have different normalization from teachers
- Easy to add new teachers with different requirements

### 3. Correctness
- Teachers receive correctly normalized inputs for their pretraining
- Student learns from properly normalized teacher outputs
- Better distillation quality

## Your v5 Config

```yaml
dataset:
  # WebDataset outputs [0,1] images
  train_dataset:
    tar_data_sources:
      - root_dir: /path/to/source_a_webdataset/1000
    
  # Student normalization (applied in model)
  augmentation:
    mean: [0.707223, 0.578729, 0.703617]  # H-optimus-0
    std: [0.211883, 0.230117, 0.177517]

distill:
  teacher:
    # Teacher normalization (applied in distiller)
    - model: h_optimus_0
      norm_mean: [0.707223, 0.578729, 0.703617]
      norm_std: [0.211883, 0.230117, 0.177517]
      
    - model: prov_gigapath
      norm_mean: [0.485, 0.456, 0.406]  # ImageNet
      norm_std: [0.229, 0.224, 0.225]
```

## Key Code Locations

| File | Function | Purpose |
|------|----------|---------|
| `dataloader/ops/fast_to_tensor.py` | `fast_to_tensor()` | Convert PIL to [0,1] tensor |
| `dataloader/data_pipeline.py` | WebDataset pipeline | Output [0,1] images |
| `distillation/distiller.py` | `_apply_teacher_normalization()` | Per-teacher normalization |
| `cv/backbone_v2/radio.py` | `forward()` | Student normalization in model |

## Summary

The statement means:

✅ **WebDataset now outputs raw [0,1] images** (not pre-normalized)

✅ **Each teacher gets its own normalization** applied in distiller

✅ **Student applies its own normalization** in the model

✅ **Everything is consistent and explicit**

This is a **good change** - it makes the normalization flow clearer and more flexible!
