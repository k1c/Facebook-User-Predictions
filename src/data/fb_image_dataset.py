from typing import List
from typing import Tuple

import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from util.image_utils import detecting_faces, expand, crop_image, resize_image, normalize_image
from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels


class FBImageDataset(Dataset):
    def __init__(self, features: List[FBUserFeatures], labels: List[FBUserLabels] or None):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image = self.features[idx].image
        label = self.labels[idx].gender

        # change label to onehot encoding
        if label == 0:
            label = [1, 0]
        else:
            label = [0, 1]

        # apply image transforms
        coordinates = detecting_faces(image)

        if coordinates is not None:
            coordinates = expand(coordinates)
            image = crop_image(image, coordinates)

        image = resize_image(image)
        image = normalize_image(image)

        if self.labels is not None:
            return torch.Tensor(image), torch.Tensor(label)
        return torch.Tensor(image)

    def __len__(self):
        return len(self.features)
