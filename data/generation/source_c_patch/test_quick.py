#!/usr/bin/env python3
"""Quick test: process only first 2 WSIs from hematologic dataset."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from patch_extraction import process_wsi

SOURCE_C_BASE = "<PATHOLOGY_DATASETS_DIR>/source_c"
OUTPUT = "<PATHOLOGY_DATASETS_DIR>/source_c_patch/test_output"

# Test on first 2 WSIs
test_wsIs = [
    f"{SOURCE_C_BASE}/source C-hematologic/case_000/slide_H&E_0.tiff",
    f"{SOURCE_C_BASE}/source C-hematologic/case_001/slide_H&E_0.tiff",
]

print(f"Testing patch extraction on {len(test_wsIs)} WSIs")
print(f"Output: {OUTPUT}\n")

for i, wsi in enumerate(test_wsIs, 1):
    print(f"\n{'='*60}")
    print(f"Test [{i}/{len(test_wsIs)}]: {os.path.basename(os.path.dirname(wsi))}")
    print(f"{'='*60}")
    
    num_patches, out_dir = process_wsi(
        wsi, 
        OUTPUT, 
        patch_size=1024, 
        min_fg_ratio=0.5
    )
    
    print(f"\nResult: {num_patches} patches saved to {out_dir}")

# Verify output
print(f"\n{'='*60}")
print("Verification")
print(f"{'='*60}")

for wsi_dir in sorted(os.listdir(OUTPUT)):
    full_path = os.path.join(OUTPUT, wsi_dir)
    if os.path.isdir(full_path):
        patches = [f for f in os.listdir(full_path) if f.endswith('.png')]
        csvs = [f for f in os.listdir(full_path) if f.endswith('.csv')]
        
        print(f"\n{wsi_dir}:")
        print(f"  PNG patches: {len(patches)}")
        print(f"  CSV files: {len(csvs)}")
        
        # Check first patch size
        if patches:
            from PIL import Image
            img = Image.open(os.path.join(full_path, patches[0]))
            print(f"  First patch size: {img.size}")
            
        # Show CSV content
        if csvs:
            csv_path = os.path.join(full_path, csvs[0])
            with open(csv_path) as f:
                lines = f.readlines()
            print(f"  CSV ({len(lines)} lines):")
            for line in lines[:4]:
                print(f"    {line.strip()}")
