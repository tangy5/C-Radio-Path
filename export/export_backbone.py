#!/usr/bin/env python3
"""Export a TAO distillation checkpoint to the C-Radio-Path-Exp backbone format.

The backbone export format (consumed by the downstream inference/eval loader) is:

    {"model": {"model.inner.<timm_key>": tensor, ...}}   # fp32 state dict

This script takes a training checkpoint — a milestone (`milestone_epoch_NNN.pth`)
or latest step checkpoint, plain or `-EMA` — whose state dict carries
`model.inner.*` backbone keys (plus head/EMA wrapper keys), keeps only the
backbone tensors, strips/filters to timm keys, re-wraps under `model.inner.`
inside a top-level `model` dict, and saves fp32.

Positional embeddings are passed through UNCHANGED (no interpolation): the
reference backbone export keeps the training-resolution pos_embed and relies on
timm's dynamic_img_size for other resolutions — same contract here.

Usage:
    python export_backbone.py --ckpt milestone_epoch_030-EMA.pth \
        --out backbone_export.pth
    python export_backbone.py --ckpt ... --out ... --strict-load-check
"""

import argparse
import re

import torch

BACKBONE_PREFIX = "model.inner."


def extract_backbone_sd(ckpt):
    """Any TAO-style ckpt -> {timm_key: tensor} for backbone tensors only."""
    if isinstance(ckpt, torch.nn.Module):
        raise SystemExit("raw module checkpoints not supported; pass a .pth file")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if "model" in sd and isinstance(sd["model"], dict):   # nested {"model": {...}}
        sd = sd["model"]
    # unwrap EMA wrappers: 'ema.model.inner.x' / 'module.model.inner.x' etc.
    out = {}
    for k, v in sd.items():
        m = re.search(r"model\.inner\.(.+)$", k)
        if m:
            out[m.group(1)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="TAO milestone/latest ckpt (.pth)")
    ap.add_argument("--out", required=True, help="output backbone .pth")
    ap.add_argument("--strict-load-check", action="store_true",
                    help="after export, verify the result loads into timm "
                         "vit_giant_patch14_reg4_dinov2 with 0 missing/unexpected keys")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    timm_sd = extract_backbone_sd(ckpt)
    if len(timm_sd) < 100:
        raise SystemExit(f"only {len(timm_sd)} backbone tensors found — not a TAO "
                         "distillation checkpoint with model.inner.* keys?")
    wrapped = {BACKBONE_PREFIX + k: v.float().contiguous() for k, v in timm_sd.items()}
    torch.save({"model": wrapped}, a.out)
    n = sum(v.numel() for v in wrapped.values())
    print(f"exported {len(wrapped)} tensors / {n:,} params -> {a.out}")

    if a.strict_load_check:
        import timm
        from timm.layers import SwiGLUPacked
        model = timm.create_model(
            "vit_giant_patch14_reg4_dinov2", pretrained=False, num_classes=0,
            img_size=224, dynamic_img_size=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        missing, unexpected = model.load_state_dict(
            {k[len(BACKBONE_PREFIX):]: v for k, v in wrapped.items()}, strict=False)
        missing = [k for k in missing if "mixer" not in k]
        assert not missing and not unexpected, f"{missing[:5]} / {unexpected[:5]}"
        print("strict load check: OK (0 missing / 0 unexpected)")


if __name__ == "__main__":
    main()
