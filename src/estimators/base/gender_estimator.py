from abc import abstractmethod
from typing import List

from data.user_features import UserFeatures
from data.user_labels import UserLabels
import pandas as pd
from estimators.base.base_estimator import BaseEstimator


class GenderEstimator(BaseEstimator):

    MAJORITY_CLASS = 1

    @abstractmethod
    def fit(self, features: List[UserFeatures], oxford_df: pd.DataFrame, labels: List[UserLabels]) -> None:
        pass

    @abstractmethod
    def predict(self, features: List[UserFeatures], oxford_df: pd.DataFrame) -> List[int]:
        pass
