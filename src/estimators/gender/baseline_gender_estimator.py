from collections import Counter
from typing import List

import pandas as pd
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.gender_estimator import GenderEstimator


class BaselineGenderEstimator(GenderEstimator):

    def __init__(self):
        self.prediction: int = None

    def fit(self, features: List[UserFeatures], oxford_df: pd.DataFrame, labels: List[UserLabels]) -> None:
        gender_counter = Counter([label.gender for label in labels])
        self.prediction = gender_counter.most_common(n=1)[0][0]

    def predict(self, features: List[UserFeatures], oxford_df: pd.DataFrame) -> List[int]:
        return [self.prediction for _ in range(len(features))]
