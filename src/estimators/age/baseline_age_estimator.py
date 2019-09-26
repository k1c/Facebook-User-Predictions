from collections import Counter
from typing import List

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from estimators.base.age_estimator import AgeEstimator


class BaselineAgeEstimator(AgeEstimator):
    def __init__(self):
        self.prediction: str = None

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        age_counter = Counter([label.age for label in labels])
        self.prediction = age_counter.most_common(n=1)[0][0]

    def predict(self, features: List[FBUserFeatures]) -> List[str]:
        return [self.prediction for _ in range(len(features))]
