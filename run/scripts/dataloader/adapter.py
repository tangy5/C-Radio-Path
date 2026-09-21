from typing import Optional

import torch
from torch.utils.data import IterableDataset
from logging import getLogger

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import SharedEpoch

_LOGGER = getLogger(__name__)


def _forever_iterator(ds: IterableDataset,
                      shared_epoch: Optional[SharedEpoch]):
    def _get_iter():
        vals = iter(ds)
        if shared_epoch is not None:
            shared_epoch.increment()
        return vals

    vals = _get_iter()
    while True:
        try:
            v = next(vals)
            yield v
        except StopIteration:
            _LOGGER.info('Dataset iterator exhausted. Fetching a new one.')
            vals = _get_iter()


class DatasetWrapper:
    """Wrapper class to provide dataset interface for IterablePrefetcher."""
    
    def __init__(self, inner_dataset, length: int = None):
        self.inner_dataset = inner_dataset
        self._length = length
    
    def __len__(self):
        """Return the length of the dataset."""
        if self._length is not None:
            return self._length
        elif hasattr(self.inner_dataset, '__len__'):
            return len(self.inner_dataset)
        elif hasattr(self.inner_dataset, 'dataset') and hasattr(self.inner_dataset.dataset, '__len__'):
            return len(self.inner_dataset.dataset)
        else:
            # For WebDataset pipelines, we need to estimate or use a default
            # This is a fallback - in practice, you should provide dataset_length
            return 10000  # Default fallback length
    
    def __getattr__(self, name):
        """Delegate attribute access to the inner dataset."""
        return getattr(self.inner_dataset, name)


class LongDatasetAdaptor:
    def __init__(self, loader: IterableDataset,
                 steps_per_epoch: int, reset_each_epoch: bool = False,
                 shared_epoch: Optional[SharedEpoch] = None,
                 batch_size: int = 32,
    ) -> None:
        self.loader = loader
        self.steps_per_epoch = steps_per_epoch
        self.reset_each_epoch = reset_each_epoch
        self.shared_epoch = shared_epoch
        self.batch_size = batch_size

        if reset_each_epoch:
            next(iter(loader))
        else:
            self._iter = _forever_iterator(self.loader, self.shared_epoch)

        # Add other attributes that TAO Lightning module might expect
        self.batch_sampler = getattr(loader.inner, 'batch_sampler', None)
        # Create a dataset wrapper that provides length information
        self.dataset = DatasetWrapper(loader.inner, steps_per_epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        if self.reset_each_epoch:
            self._iter = iter(self.loader)

        for _ in range(self.steps_per_epoch):
            v = next(self._iter)
            yield v
