from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.age_estimator import AgeEstimator

from sklearn.utils import shuffle


class GBDTAgeEstimator(AgeEstimator):

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=25,
            max_depth=3
        )


# get age targets for each user in liwc and nrc
    def _extract_targets(self, df: pd.DataFrame, labels: List[UserLabels]) -> np.array:
        user_id_to_label = {label.user_id: label.age for label in labels}
        targets = list()
        for _, row in df.iterrows():
            targets.append(user_id_to_label[row["userId"]])
        return targets

# get the features related to a user in liwc and nrc
    def _extract_features(self, df: pd.DataFrame, user_id: str) -> Optional[pd.DataFrame]:
        rows = df[df["userId"] == user_id]
        if len(rows) == 0:
            return None
        elif len(rows) == 1:
            return rows.to_frame().T if isinstance(rows, pd.Series) else rows
        else:
            row = rows.iloc[0]
            return row.to_frame().T if isinstance(row, pd.Series) else row

    def _merge_features(self, liwc_df: pd.DataFrame, nrc_df: pd.DataFrame):

        # remove duplicate userId and keep the first one (didn't find any duplicated for both features, but doing this in case)
        liwc_df = liwc_df.drop_duplicates(
            subset=["userId"],
            keep="first"
        )
        nrc_df = nrc_df.drop_duplicates(
            subset=["userId"],
            keep="first"
        )

        liwc_nrc_df = liwc_df.merge(nrc_df, how='inner', left_on='userId', right_on='userId',validate='one_to_one')

        return liwc_nrc_df

    def fit(self,
            features: List[UserFeatures],
            liwc_df: pd.DataFrame,
            nrc_df: pd.DataFrame,
            labels: List[UserLabels]) -> None:

        liwc_nrc_df = self._merge_features(liwc_df, nrc_df)
        liwc_nrc_features = liwc_nrc_df.drop(["userId"], axis=1)


        liwc_nrc_targets = self._extract_targets(liwc_nrc_df, labels)
        X, y = shuffle(liwc_nrc_features.values, liwc_nrc_targets)
        self.model.fit(X, y)

    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame) -> List[str]:

        age_predictions = list()
        liwc_nrc_df = self._merge_features(liwc_df, nrc_df)

        for feature in features:
            liwc_nrc_feature = self._extract_features(liwc_nrc_df, feature.user_id) # get the features for one specific user
            if liwc_nrc_feature is None:
                age_predictions.append(str(AgeEstimator.AGE_BASELINE)) # predict majority count
            else:
                liwc_nrc_feature = liwc_nrc_feature.drop(["userId"], axis=1)
                age_predictions.append(str(self.model.predict(liwc_nrc_feature)))

        return age_predictions

