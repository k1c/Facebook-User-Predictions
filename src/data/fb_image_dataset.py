from typing import List
from typing import Tuple

import numpy as np
from torch.utils.data.dataset import Dataset

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels


class FBImageDataset(Dataset):
    def __init__(self, features: List[FBUserFeatures], labels: List[FBUserLabels]):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.features[idx].image, self.labels[idx].gender

    def __len__(self):
        return len(self.features)
