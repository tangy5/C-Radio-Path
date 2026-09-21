# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for data processing modules."""

# Comprehensive list of all supported image file extensions across the codebase
SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# WebDataset image extensions string (semicolon-separated for wds.to_tuple)
# Derived from SUPPORTED_IMAGE_EXTENSIONS - using core extensions that WebDataset supports
WDS_IMAGE_EXTENSIONS_STR = ";".join(ext.lstrip('.') for ext in SUPPORTED_IMAGE_EXTENSIONS)
