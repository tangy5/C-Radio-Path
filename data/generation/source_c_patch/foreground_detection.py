"""Foreground detection module for WSI tissue segmentation.

Uses Otsu's thresholding on downsampled grayscale image with morphological
operations to create a binary tissue mask. This approach is inspired by
standard WSI preprocessing pipelines (e.g., CLAM, Histo-Miner).

Reference: 
- "Efficient Tissue Detection in Whole-Slide Images Using Classical Methods"
- Otsu on luminance channel with morphological cleanup
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

# Disable PIL decompression bomb check for large WSI
Image.MAX_IMAGE_PIXELS = None


def get_thumbnail_from_tiff(tiff_path: str, downsample: int = 32) -> np.ndarray:
    """Load a downsampled thumbnail from a pyramid TIFF.
    
    Args:
        tiff_path: Path to pyramid TIFF file
        downsample: Target downsample factor
        
    Returns:
        RGB thumbnail as numpy array (H, W, 3)
    """
    img = Image.open(tiff_path)
    n_levels = img.n_frames if hasattr(img, 'n_frames') else 1
    
    # Find best level for target downsample
    level_dims = []
    for i in range(n_levels):
        img.seek(i)
        level_dims.append((img.width, img.height))
    
    width_0, height_0 = level_dims[0]
    target_w = width_0 // downsample
    target_h = height_0 // downsample
    
    # Select level closest to target size
    best_level = 0
    best_diff = float('inf')
    for i, (w, h) in enumerate(level_dims):
        diff = abs(w - target_w) + abs(h - target_h)
        if diff < best_diff:
            best_diff = diff
            best_level = i
    
    # Read the selected level
    img.seek(best_level)
    
    # If still too large, resize further
    thumb = img.copy()
    if thumb.width > target_w * 2 or thumb.height > target_h * 2:
        thumb = thumb.resize((target_w, target_h), Image.LANCZOS)
    
    return np.array(thumb.convert('RGB'))


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB image.
    
    Uses standard Rec. 601 luma coefficients.
    """
    gray = 0.299 * rgb[:, :, 0].astype(np.float64) + \
           0.587 * rgb[:, :, 1].astype(np.float64) + \
           0.114 * rgb[:, :, 2].astype(np.float64)
    return gray.astype(np.uint8)


def otsu_foreground_mask(tiff_path: str, downsample: int = 32,
                         morph_close_k: int = 15, morph_open_k: int = 5) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Generate binary foreground mask for WSI tissue detection.
    
    Pipeline:
    1. Load downsampled thumbnail
    2. Convert to luminance (grayscale)
    3. Apply Otsu's thresholding
    4. Invert (tissue is dark, background is bright white)
    5. Morphological opening to remove noise
    6. Morphological closing to fill holes in tissue
    
    Args:
        tiff_path: Path to pyramid TIFF WSI
        downsample: Downsample factor for processing
        morph_close_k: Closing kernel size
        morph_open_k: Opening kernel size
        
    Returns:
        Tuple of (binary mask at downsampled resolution, original dimensions (w, h))
    """
    # Step 1: Get thumbnail
    thumb_rgb = get_thumbnail_from_tiff(tiff_path, downsample)
    orig_h, orig_w = thumb_rgb.shape[:2]
    
    # Step 2: Convert to luminance
    gray = compute_luminance(thumb_rgb)
    
    # Step 3: Otsu thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Step 4: Morphological operations
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_k, morph_close_k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_k, morph_open_k))
    
    # Opening first to remove small noise, then closing to fill tissue holes
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    # Get original WSI dimensions
    img = Image.open(tiff_path)
    img.seek(0)  # Level 0 = highest resolution
    orig_wsi_w, orig_wsi_h = img.width, img.height
    
    # Calculate actual scale factor
    scale_x = orig_wsi_w / orig_w
    scale_y = orig_wsi_h / orig_h
    
    return mask, (orig_wsi_w, orig_wsi_h), (scale_x, scale_y)


def upscale_mask_to_level0(mask: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
    """Upscale the binary mask to level 0 (full resolution).
    
    Args:
        mask: Binary mask at downsampled resolution
        scale: (scale_x, scale_y) from downsampled to level 0
        
    Returns:
        Binary mask at level 0 resolution
    """
    target_h = int(mask.shape[0] * scale[1])
    target_w = int(mask.shape[1] * scale[0])
    
    upscaled = cv2.resize(mask.astype(np.uint8), (target_w, target_h), 
                         interpolation=cv2.INTER_NEAREST)
    return upscaled > 0


def compute_foreground_contours(mask: np.ndarray) -> list:
    """Extract contours from foreground mask.
    
    Args:
        mask: Binary mask (2D bool or uint8 array)
        
    Returns:
        List of contours (for visualization or filtering)
    """
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == "__main__":
    import time
    
    # Test on a sample file
    tiff_path = "<PATHOLOGY_DATASETS_DIR>/source_c/source C-hematologic/case_000/slide_H&E_0.tiff"
    
    print(f"Processing: {tiff_path}")
    start = time.time()
    
    mask, orig_dims, scale = otsu_foreground_mask(tiff_path, downsample=32)
    elapsed = time.time() - start
    
    print(f"Mask shape (downsampled): {mask.shape}")
    print(f"Original WSI dimensions: {orig_dims}")
    print(f"Scale factors: {scale}")
    print(f"Foreground pixels: {(mask > 0).sum()} / {mask.size} ({100*(mask > 0).sum()/mask.size:.1f}%)")
    print(f"Time: {elapsed:.2f}s")
    
    # Save mask for visualization
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite("/tmp/test_foreground_mask.png", mask_vis)
    print("Saved mask to /tmp/test_foreground_mask.png")
