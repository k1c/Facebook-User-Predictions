from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.gender_estimator import GenderEstimator

from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle


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

    def _extract_oxford_features(self, oxford_df: pd.DataFrame, user_id: str) -> Optional[pd.DataFrame]:
        rows = oxford_df[oxford_df["userId"] == user_id]
        if len(rows) == 0:
            return None
        elif len(rows) == 1:

            return rows.to_frame().T if isinstance(rows, pd.Series) else rows
        else:
            row = rows.iloc[0]
            return row.to_frame().T if isinstance(row, pd.Series) else row

    def fit(self, features: List[UserFeatures], oxford_df: pd.DataFrame, labels: List[UserLabels]) -> None:

        # Sometimes we have two rows with the same userId. There might be two different people of different
        # genders in this, and this will only confuse our algorithm. So we drop duplicates.
        oxford_df = oxford_df.drop_duplicates(
            subset=["userId"],
            keep="first"
        )
        features = oxford_df.drop(
            ["userId", "faceID"],
            axis=1
        )
        targets = self._extract_targets(oxford_df, labels)
        smote = SMOTE()
        X, y = shuffle(features.values, targets)
        X_res, y_res = smote.fit_resample(X, y)
        self.model.fit(X_res, y_res)

    def predict(self, features: List[UserFeatures], oxford_df: pd.DataFrame) -> List[int]:
        predictions = list()
        for feature in features:
            oxford_feature = self._extract_oxford_features(oxford_df, feature.user_id)
            if oxford_feature is None:
                predictions.append(GenderEstimator.MAJORITY_CLASS)
            else:
                predictions.append(
                    int(self.model.predict(
                        oxford_feature.drop(
                            ["userId", "faceID"],
                            axis=1
                        )
                    ).item())
                )
        return predictions
