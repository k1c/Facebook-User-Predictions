from collections import Counter
from typing import List

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.age_estimator import AgeEstimator


class BaselineAgeEstimator(AgeEstimator):
    def __init__(self):
        self.prediction: str = None

    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        age_counter = Counter([label.age for label in labels])
        self.prediction = age_counter.most_common(n=1)[0][0]

    def predict(self, features: List[UserFeatures]) -> List[str]:
        return [self.prediction for _ in range(len(features))]
