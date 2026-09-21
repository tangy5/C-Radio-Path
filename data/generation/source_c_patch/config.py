"""Configuration for source C patch extraction pipeline."""

import os

# Base paths
SOURCE_C_BASE = "<PATHOLOGY_DATASETS_DIR>/source_c"
OUTPUT_BASE = "<PATHOLOGY_DATASETS_DIR>/source_c_patch"

# Datasets to process (in order)
DATASETS = [
    "source C-metadata",       # Skip - no images
    "source C-hematologic",
    "source C-gastrointestinal",
    "source C-colorectal-b2",
    "source C-thorax",
    "source C-breast",
    "source C-colorectal-b1",
    "source C-skin-b1",
    "source C-skin-b2",
]

# Patch extraction settings
PATCH_SIZE = 1024           # Patch size in pixels (at level 0)
PATCH_FORMAT = "jpeg"       # Output image format (jpeg saves ~60% space vs png at q=95)

# Foreground detection settings
OTSU_DOWNSAMPLE_FACTOR = 32  # Downsample factor for Otsu thresholding (speed)
MORPH_CLOSE_KERNEL = 15      # Morphological closing kernel size (remove holes)
MORPH_OPEN_KERNEL = 5        # Morphological opening kernel size (remove noise)
MIN_FOREGROUND_RATIO = 0.5   # Minimum foreground ratio to keep a patch

# Sliding window settings
OVERLAP = 0                  # No overlap by default (can adjust)
EDGE_PADDING = False         # Whether to include partial edge patches

# Output settings
SAVE_MASKS = False           # Save foreground masks (debug)
SAVE_COORDINATES = True      # Save coordinate CSV for each WSI
MAX_WORKERS = 100            # Parallel workers (128 vCPUs available, leave 28 for system/I/O)
JPEG_QUALITY = 95            # JPEG quality (95 = visually lossless, ~60% smaller than PNG)

# Level to extract from (0 = highest resolution)
EXTRACTION_LEVEL = 0

# Case selection file for balanced sampling (optional)
# Set to None to process all cases
CASE_SELECTION_FILE = os.path.join(os.path.dirname(__file__), "selected_cases.json")
