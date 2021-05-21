from typing import Iterator
from nuscenes.nuscenes import NuScenes
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import argoverse
import torch.utils.data as thd
import numpy as np
from torch.utils.data.dataset import T_co

"""

Core toch datasets and dataloaders for combining multiple datasets together. Each source will have its own dataset
that converta outputs to all the same format, then yields it to the training process. 

"""


class AutomotiveDataset(thd.IterableDataset):
    """Takes a list of different datasets for each of the sources, and randomly samples from each of them, based on the config"""

    def __init__(self, config, *datasets):
        self.datasets = datasets
        assert all([isinstance(d, thd.IterableDataset) for d in
                    self.datasets]), "AutomotiveDataset only supports IterableDataset"
        self.config = config
        # Fraction of each dataset to use, e.g. if we want more NuScenes or Waymo, etc.
        self.fractions = np.asarray(config.get("fractions", [1 for _ in range(len(self.datasets))]))
        self.fractions /= np.sum(self.fractions)
        self.counts = 100 * np.ones(
            len(self.datasets))  # Actual count of how many examples per dataset, sets to 100 so first examples don't massively change the percentages

    def __iter__(self) -> Iterator[T_co]:
        while True:
            self.datasets = np.random.sh
            for i, d in enumerate(self.datasets):
                for x in d:
                    if self.counts[i] / np.sum(self.counts) > 1.1 * self.fractions[i]:  # 10% leeway
                        yield x
                        self.counts[i] += 1
                    else:
                        break  # Escape to next dataset

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, thd.IterableDataset), "AutomotiveDataset only supports IterableDataset"
            # Cannot verify that all self.datasets are Sized
            total += len(d)  # type: ignore
        return total


class NuScenesDataset(thd.IterableDataset):

    def __init__(self, config):
        self.config = config
        self.root_dir = config.get("root_dir", "")
        self.version = config.get("", "v1.0")
        self.type = config.get("task_type", "annotation")
        self.loader = NuScenes(version=self.version, dataroot=self.root_dir)
        self.classes = config.get("classes", ["vehicle"])

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield 0


class ArgoDataset(thd.IterableDataset):

    def __init__(self, config):
        self.config = config
        self.root_dir = config.get("root_dir", "")
        self.loader = ArgoverseTrackingLoader(self.root_dir)
        self.log_list = self.loader.log_list
        self.type = config.get("task_type", "annotation")
        self.classes = config.get("classes", ["vehicle"])

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.log_list))
            scene = self.loader.get(self.log_list[idx])

            yield scene


class A2D2Dataset(thd.IterableDataset):

    def __init__(self, config):
        self.config = config
        self.root_dir = config.get("root_dir", "")
        self.type = config.get("task_type", "annotation")
        self.classes = config.get("classes", ["vehicle"])

    def __iter__(self):
        pass


class WaymoDataset(thd.IterableDataset):

    def __init__(self, config):
        self.config = config
        self.root_dir = config.get("root_dir", "")
        self.type = config.get("task_type", "annotation")
        self.classes = config.get("classes", ["vehicle"])

    def __iter__(self):
        pass
