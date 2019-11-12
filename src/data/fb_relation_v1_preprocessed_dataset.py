from typing import Tuple
import torch.Tensor
import numpy as np
from torch.utils.data.dataset import Dataset


class FBRelationV1PreprocessedDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor or None):
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
