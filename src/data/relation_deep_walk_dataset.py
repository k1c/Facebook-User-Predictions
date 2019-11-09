from typing import Tuple
import torch.tensor
import numpy as np
from torch.utils.data.dataset import Dataset


class RelationDeepWalkDataset(Dataset):
    def __init__(self, features: torch.tensor, labels: torch.tensor or None):
        if labels is not None:
            assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)
