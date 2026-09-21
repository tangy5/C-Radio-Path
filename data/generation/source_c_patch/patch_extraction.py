"""Patch extraction module for WSI.

Generates sliding window patches from foreground regions and extracts them
from the full-resolution WSI. Outputs PNG patches with coordinate metadata.
"""

import os
import csv
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

# Disable PIL decompression bomb check
Image.MAX_IMAGE_PIXELS = None


def generate_patch_coordinates(mask: np.ndarray, patch_size: int = 1024,
                                scale: Tuple[float, float] = (1.0, 1.0),
                                min_fg_ratio: float = 0.5,
                                overlap: int = 0) -> List[Dict]:
    """Generate patch coordinates from foreground mask using sliding window.
    
    Works entirely in mask space for efficiency, then converts to level 0 coords.
    
    Args:
        mask: Binary foreground mask at downsampled resolution
        patch_size: Size of patches at level 0 (exact pixels to crop)
        scale: (scale_x, scale_y) from mask to level 0
        min_fg_ratio: Minimum foreground pixel ratio to keep patch
        overlap: Overlap between adjacent patches (0 = no overlap)
        
    Returns:
        List of dicts with keys: patch_idx, x, y, w, h, fg_ratio
    """
    # Patch size in mask coordinates
    patch_h_mask = max(1, int(patch_size / scale[1]))
    patch_w_mask = max(1, int(patch_size / scale[0]))
    
    mask_h, mask_w = mask.shape
    
    # Stride in mask coordinates (at least 1 pixel)
    stride_mask = max(1, int(patch_size * (1 - overlap / 100) / scale[0]))
    
    patches = []
    patch_idx = 0
    
    # Binary mask for fast counting
    fg_mask = (mask > 0).astype(np.uint8)
    patch_area_mask = patch_h_mask * patch_w_mask
    
    # Iterate in mask space
    for y_m in range(0, mask_h - patch_h_mask + 1, stride_mask):
        for x_m in range(0, mask_w - patch_w_mask + 1, stride_mask):
            # Check foreground ratio
            region = fg_mask[y_m:y_m+patch_h_mask, x_m:x_m+patch_w_mask]
            fg_count = np.count_nonzero(region)
            fg_ratio = fg_count / patch_area_mask if patch_area_mask > 0 else 0
            
            if fg_ratio >= min_fg_ratio:
                # Convert mask coords to level 0
                x_0 = int(x_m * scale[0])
                y_0 = int(y_m * scale[1])
                
                patches.append({
                    'patch_idx': patch_idx,
                    'x': x_0,
                    'y': y_0,
                    'w': patch_size,
                    'h': patch_size,
                    'fg_ratio': round(fg_ratio, 4)
                })
                patch_idx += 1
    
    return patches


def extract_patch_from_tiff(tiff_path: str, x: int, y: int, 
                             w: int, h: int) -> Optional[np.ndarray]:
    """Extract a region from pyramid TIFF at level 0.
    
    Args:
        tiff_path: Path to pyramid TIFF
        x, y: Top-left coordinates at level 0
        w, h: Width and height at level 0
        
    Returns:
        RGB patch as numpy array, or None if extraction fails
    """
    img = Image.open(tiff_path)
    img.seek(0)  # Level 0 = full resolution
    
    # Ensure we don't go outside image bounds
    img_w, img_h = img.width, img.height
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return None
    
    # Crop the region
    crop_box = (x, y, x + w, y + h)
    patch = img.crop(crop_box)
    
    # Convert to RGB if necessary
    if patch.mode != 'RGB':
        patch = patch.convert('RGB')
    
    return np.array(patch)


def extract_all_patches(tiff_path: str, patches: List[Dict],
                         output_dir: str, wsi_name: str,
                         patch_size: int = 1024,
                         save_coords: bool = True,
                         patch_format: str = "jpeg",
                         jpeg_quality: int = 95) -> str:
    """Extract all patches and save them as image files.
    
    Args:
        tiff_path: Path to source WSI TIFF
        patches: List of patch coordinate dicts
        output_dir: Directory to save patches
        wsi_name: Name of WSI (for naming patches)
        patch_size: Patch size in pixels
        save_coords: Whether to save coordinate CSV
        patch_format: Output format ("jpeg" or "png")
        jpeg_quality: JPEG quality (1-100), only used if patch_format="jpeg"
        
    Returns:
        Path to coordinate CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    coords_file = os.path.join(output_dir, f"{wsi_name}_patches.csv")
    
    img = Image.open(tiff_path)
    img.seek(0)
    img_w, img_h = img.width, img.height
    
    ext = ".jpg" if patch_format.lower() in ("jpeg", "jpg") else ".png"
    pil_format = "JPEG" if ext == ".jpg" else "PNG"
    
    extracted = 0
    failed = 0
    coord_rows = []
    
    start = time.time()
    
    for i, patch_info in enumerate(patches):
        x, y, w, h = patch_info['x'], patch_info['y'], patch_info['w'], patch_info['h']
        fg_ratio = patch_info['fg_ratio']
        
        # Clamp to image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            failed += 1
            continue
        
        try:
            # Crop the exact region at original resolution - NO resizing
            crop_box = (x, y, x + w, y + h)
            patch = img.crop(crop_box)
            
            # Convert to RGB if needed, but NEVER resize
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            
            # Generate patch filename with actual dimensions
            actual_w, actual_h = patch.size
            patch_name = f"{wsi_name}_x{x}_y{y}_w{actual_w}_h{actual_h}_fg{fg_ratio:.2f}{ext}"
            patch_path = os.path.join(output_dir, patch_name)
            
            if pil_format == "JPEG":
                patch.save(patch_path, "JPEG", quality=jpeg_quality)
            else:
                patch.save(patch_path, "PNG")
            
            coord_rows.append({
                'patch_name': patch_name,
                'x': x,
                'y': y,
                'w': actual_w,
                'h': actual_h,
                'fg_ratio': fg_ratio
            })
            extracted += 1
            
        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first few errors
                print(f"    Warning: Failed to extract patch {i}: {e}")
    
    elapsed = time.time() - start
    
    # Save coordinate file
    if save_coords and coord_rows:
        with open(coords_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['patch_name', 'x', 'y', 'w', 'h', 'fg_ratio'])
            writer.writeheader()
            writer.writerows(coord_rows)
    
    print(f"    Extracted {extracted} patches, {failed} failed in {elapsed:.1f}s")
    
    return coords_file


def process_wsi(tiff_path: str, output_base: str, patch_size: int = 1024,
                min_fg_ratio: float = 0.5, overlap: int = 0,
                save_coords: bool = True,
                patch_format: str = "jpeg",
                jpeg_quality: int = 95) -> Tuple[int, str]:
    """Process a single WSI: detect foreground and extract patches.
    
    Args:
        tiff_path: Path to WSI TIFF file
        output_base: Base output directory
        patch_size: Patch size at level 0
        min_fg_ratio: Minimum foreground ratio for patches
        overlap: Overlap percentage
        save_coords: Save coordinate CSV
        patch_format: Output format ("jpeg" or "png")
        jpeg_quality: JPEG quality (1-100)
        
    Returns:
        Tuple of (number of patches extracted, output directory)
    """
    from foreground_detection import otsu_foreground_mask
    
    # Get WSI name from path
    # Path pattern: .../dataset/case_XXX/slide_H&E_0.tiff
    case_dir = os.path.basename(os.path.dirname(tiff_path))
    wsi_stem = Path(tiff_path).stem  # e.g., "slide_H&E_0"
    wsi_name = f"{case_dir}_{wsi_stem}"
    
    # Output directory for this WSI
    wsi_output_dir = os.path.join(output_base, wsi_name)
    
    # Resume check: look for existing patches (png or jpg) or CSV
    if os.path.exists(wsi_output_dir):
        existing_patches = [f for f in os.listdir(wsi_output_dir) 
                           if f.endswith('.png') or f.endswith('.jpg')]
        existing_csvs = [f for f in os.listdir(wsi_output_dir) if f.endswith('_patches.csv')]
        
        if existing_patches or existing_csvs:
            n_patches = len(existing_patches)
            csv_info = f", CSV with {len(existing_csvs)} file(s)" if existing_csvs else ""
            print(f"  RESUME: Already processed ({n_patches} patches{csv_info})")
            return n_patches, wsi_output_dir
    
    print(f"  Processing: {tiff_path}")
    print(f"  WSI name: {wsi_name}")
    
    # Step 1: Foreground detection
    start_fg = time.time()
    mask, orig_dims, scale = otsu_foreground_mask(tiff_path)
    elapsed_fg = time.time() - start_fg
    
    fg_pixels = np.count_nonzero(mask)
    fg_pct = 100 * fg_pixels / mask.size
    print(f"  Foreground: {fg_pct:.1f}% (mask: {mask.shape}, scale: {scale[0]:.1f}x)")
    print(f"  Foreground detection time: {elapsed_fg:.2f}s")
    
    # Step 2: Generate patch coordinates
    patches = generate_patch_coordinates(
        mask, 
        patch_size=patch_size,
        scale=scale,
        min_fg_ratio=min_fg_ratio,
        overlap=overlap
    )
    print(f"  Generated {len(patches)} candidate patches")
    
    # Step 3: Extract patches
    if patches:
        extract_all_patches(tiff_path, patches, wsi_output_dir, wsi_name, patch_size, save_coords,
                           patch_format=patch_format, jpeg_quality=jpeg_quality)
    
    return len(patches), wsi_output_dir


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Test on single WSI
    tiff = "<PATHOLOGY_DATASETS_DIR>/source_c/source C-hematologic/case_000/slide_H&E_0.tiff"
    output = "<PATHOLOGY_DATASETS_DIR>/source_c_patch/test_output"
    
    num_patches, out_dir = process_wsi(tiff, output, patch_size=1024, min_fg_ratio=0.5)
    print(f"\nDone! Extracted {num_patches} patches to {out_dir}")
    
    # List first few patches
    if os.path.exists(out_dir):
        patches = [f for f in os.listdir(out_dir) if f.endswith(('.png', '.jpg'))][:5]
        print("Sample patches:")
        for p in patches:
            print(f"  {p}")
