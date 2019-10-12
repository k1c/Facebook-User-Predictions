from abc import abstractmethod
from typing import List

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.base_estimator import BaseEstimator


class GenderEstimator(BaseEstimator):

    @abstractmethod
    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        pass

    @abstractmethod
    def predict(self, features: List[UserFeatures]) -> List[int]:
        pass
