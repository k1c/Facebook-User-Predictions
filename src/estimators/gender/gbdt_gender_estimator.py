from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.gender_estimator import GenderEstimator


class GBDTGenderEstimator(GenderEstimator):
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3
        )

    def _extract_targets(self, oxford_df: pd.DataFrame, labels: List[UserLabels]) -> np.array:
        user_id_to_gender = {label.user_id: label.gender for label in labels}
        targets = list()
        for _, row in oxford_df.iterrows():
            targets.append(user_id_to_gender[row["userId"]])
        return targets

    def fit(self, features: List[UserFeatures], oxford_df: pd.DataFrame, labels: List[UserLabels]) -> None:

        features = oxford_df.drop(
            ["userId", "faceID"],
            axis=1
        )
        targets = self._extract_targets(oxford_df, labels)
        self.model.fit(features.values, targets)

    def predict(self, features: List[UserFeatures], oxford_df: pd.DataFrame) -> List[int]:
        predictions = list()
        for _, row in oxford_df.iterrows():
            predictions.append(
                self.model.predict(
                    row.drop(
                        ["userID", "faceID"],
                        axis=1
                    )
                )
            )
        return predictions
