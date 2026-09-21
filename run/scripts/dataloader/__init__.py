# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Classification dataloader module."""
from pathlib import Path
from typing import Optional, Union, Dict, Any

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import CLDataset


def build_dataset(
    images_dir: Optional[Union[str, Path]] = None,
    augmentation: Optional[Dict[str, Any]] = None,
    image_size: Optional[int] = None,
    nolabel_folder: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> CLDataset:
    """Build a CLDataset from an image directory.

    The WebDataset (tar) path is handled directly by
    ``CLDataModule._build_wds_pipeline()`` and does not go through this
    function.

    Args:
        images_dir: Directory containing images.
        augmentation: Augmentation configuration dict.
        image_size: Target image size.
        nolabel_folder: Path to image folder with no labels.
        root_dir: Root directory for the dataset.

    Returns:
        A CLDataset instance.
    """
    if not images_dir:
        raise ValueError("images_dir must be provided for CLDataset.")
    return CLDataset(
        root_dir=root_dir,
        data_path=images_dir,
        augmentation=augmentation,
        nolabel_folder=nolabel_folder,
        img_size=image_size,
        to_tensor=True,
    )
