from abc import abstractmethod
from typing import List

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.base_estimator import BaseEstimator


class PersonalityEstimator(BaseEstimator):

    @abstractmethod
    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        pass

    @abstractmethod
    def predict(self, features: List[UserFeatures]) -> List[PersonalityTraits]:
        pass
