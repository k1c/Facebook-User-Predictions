from abc import ABC
from typing import List, Tuple

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels


class BaseEstimator(ABC):
    @staticmethod
    def train_valid_split(features: List[FBUserFeatures],
                          labels: List[FBUserLabels],
                          valid_split: float) -> Tuple[List[FBUserFeatures], List[FBUserLabels],
                                                       List[FBUserFeatures], List[FBUserLabels]]:
        train_until_idx = int(round(valid_split * len(features)))
        # always choose the first x elements for training.
        train_features, train_labels = features[:train_until_idx], labels[:train_until_idx]
        valid_features, valid_labels = features[train_until_idx:], labels[train_until_idx:]
        return train_features, train_labels, valid_features, valid_labels
