from abc import ABC
from typing import List, Tuple

from data.user_features import UserFeatures
from data.user_labels import UserLabels


class BaseEstimator(ABC):
    @staticmethod
    def train_valid_split(features: List[UserFeatures],
                          labels: List[UserLabels],
                          valid_split: float) -> Tuple[List[UserFeatures], List[UserLabels],
                                                       List[UserFeatures], List[UserLabels]]:
        train_until_idx = int(round(valid_split * len(features)))
        # always choose the first x elements for training.
        train_features, train_labels = features[:train_until_idx], labels[:train_until_idx]
        valid_features, valid_labels = features[train_until_idx:], labels[train_until_idx:]
        return train_features, train_labels, valid_features, valid_labels
