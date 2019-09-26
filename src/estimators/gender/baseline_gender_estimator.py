from collections import Counter
from typing import List

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from estimators.base.gender_estimator import GenderEstimator


class BaselineGenderEstimator(GenderEstimator):

    def __init__(self):
        self.prediction: int = None

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        gender_counter = Counter([label.gender for label in labels])
        self.prediction = gender_counter.most_common(n=1)[0][0]

    def predict(self, features: List[FBUserFeatures]) -> List[int]:
        return [self.prediction for _ in range(len(features))]
