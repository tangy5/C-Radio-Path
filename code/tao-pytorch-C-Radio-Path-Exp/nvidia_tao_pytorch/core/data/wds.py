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

"""NVDINOv2 Dataset"""

import io
from typing import Optional, Union, List, Dict, Any, Callable, Iterator
import random
from pathlib import Path
import webdataset as wds
from PIL import Image
import numpy as np
from torch.utils.data import IterableDataset
from webdataset.shardlists import expand_urls
import multiprocessing as mp

from nvidia_tao_pytorch.core.tlt_logging import logging

# Shared memory for epoch (works across spawned processes)
# The epoch info is needed to be used in ResumableShardList to avoid the same urls being used in the next epoch.
_shared_epoch = None


def _get_shared_epoch() -> int:
    """Get the shared epoch value.

    Returns:
        int: The current epoch value from shared memory.
    """
    global _shared_epoch  # pylint: disable=global-statement
    if _shared_epoch is None:
        # Initialize shared memory
        _shared_epoch = mp.Value('i', 0)
    return _shared_epoch.value


def _set_shared_epoch(epoch: int) -> None:
    """Set the shared epoch value.

    Args:
        epoch (int): The epoch number to set in shared memory.
    """
    global _shared_epoch  # pylint: disable=global-statement
    if _shared_epoch is None:
        # Initialize shared memory
        _shared_epoch = mp.Value('i', 0)
    _shared_epoch.value = epoch


class ResumableShardList(IterableDataset):
    """An iterable dataset yielding a list of URLs for webdataset shards.

    This class provides resumable iteration over webdataset shards with proper
    epoch management and distributed data parallel support.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,abstract-method
    def __init__(
        self,
        urls: Optional[Union[List[str], str]] = None,
        root: Optional[Union[str, Path]] = None,
        samples_per_file: int = 10000,
        batch_size: int = 32,
        seed: int = 42
    ):
        """Initialize the ResumableShardList with URLs or root directory.

        Args:
            urls (Optional[Union[List[str], str]]): A list of URLs as a Python list or
                brace notation string, or a file path containing URLs. Must be set if root is not provided.
            root (Optional[Union[str, Path]]): Root directory for the shards. Must be set if urls is not provided.
            samples_per_file (int): Number of samples per file. Defaults to 10000.
            batch_size (int): Total batch size (batch size per GPU * number of GPUs). Defaults to 32.
            seed (int): Random seed for shuffling. Defaults to 42.

        Raises:
            AssertionError: If both urls and root are None, or if both are provided.
            AssertionError: If URLs list contains non-string items.
        """
        super().__init__()

        assert urls is not None or root is not None, "urls and root cannot be both None"

        if urls is not None:
            if isinstance(urls, str):
                with open(urls, encoding='utf-8') as stream:
                    urls = [
                        str(Path(root) / line.strip())
                        for line in stream
                        if line.strip()
                    ]

            self.urls = expand_urls(urls)
        else:
            self.urls = [str(i) for i in Path(root).rglob("*.tar")]

        assert all(isinstance(url, str) for url in self.urls), "All URLs must be strings"

        logging.info("Found %d shards", len(self.urls))

        self.samples_per_file = samples_per_file
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self) -> int:
        """Return the number of shards in the dataset.

        Returns:
            int: The total number of shards in the dataset.
        """
        return len(self.urls)

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Return an iterator over the shards.

        Returns:
            Iterator[Dict[str, str]]: An iterator yielding dictionaries with 'url' keys.
        """
        urls = self.urls.copy()

        urls.sort()
        # Since the webdataset would set the .with_length() method which ends current epoch earlier
        # and then start the next epoch, we need to shuffle the urls accordingly to avoid the same urls being used in the next epoch.
        current_epoch = self._get_current_epoch()
        logging.info(f"wds epoch: {current_epoch}. Please check whether this epoch is identical to current lightning training epoch.")
        random.Random(self.seed + current_epoch).shuffle(urls)

        # The following two splitting methods are directly applied in the ResumableShardList class, which result in the correct ddp splitting behavior.
        urls = list(wds.split_by_node(wds.split_by_worker(urls)))

        for url in urls:
            yield {"url": url}

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for this instance.

        This is the clean public interface for setting epoch.
        It handles all the internal epoch setting logic.

        Args:
            epoch (int): The current epoch number.
        """
        _set_shared_epoch(epoch)

    def _get_current_epoch(self) -> int:
        """Get current epoch using multiple fallback methods.

        Returns:
            int: The current epoch value from shared memory.
        """
        shared_epoch = _get_shared_epoch()
        return shared_epoch


def pil_loader_with_empty_support(key: str, data: bytes) -> Optional[Image.Image]:
    """Load an image from bytes data if it has a valid extension, otherwise return None.

    Args:
        key (str): The file extension or key identifying the image type.
        data (bytes): The raw image data in bytes.

    Returns:
        Optional[Image.Image]: PIL Image object if valid image, None if invalid extension,
            or a black image if the data is corrupted.
    """
    if key.strip() not in {".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".pbm", ".pnm"}:
        return None
    try:
        with io.BytesIO(data) as stream:
            img = Image.open(stream).convert("RGB")
    except Exception:  # pylint: disable=broad-exception-caught
        # In case of corrupt sample, just return a blank image
        # Images have already been filtered before creating the webdata tars.
        # So, this part should not be reached.
        # However, for some unavoidable cases, if this is reached, lets not kill the job.
        print("Corrupt sample read")
        img = Image.new("RGB", (256, 256), (0, 0, 0))
    # img = np.array(img)
    return img


def has_image(sample: Dict[str, Any]) -> bool:
    """Check if sample contains an image with valid extension.

    Args:
        sample (Dict[str, Any]): A dictionary containing sample data with keys as file extensions.

    Returns:
        bool: True if the sample contains an image with a valid extension, False otherwise.
    """
    image_extensions = {"jpg", "jpeg", "png", "ppm", "pgm", "pbm", "pnm"}
    return bool(image_extensions.intersection(sample.keys()))


# Example can be found in the following files.
# Note that the epoch setting logic is needed to be added to the pl.LightningDataModule and pl.LightningModule.
# nvidia_tao_pytorch/ssl/nvdinov2/dataloader/dataset_wds.py
# nvidia_tao_pytorch/ssl/nvdinov2/dataloader/pl_dinov2_data_module.py
# nvidia_tao_pytorch/ssl/nvdinov2/model/pl_model.py
