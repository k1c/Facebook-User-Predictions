from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.personality_estimator import PersonalityEstimator

from sklearn.utils import shuffle


class GBDTPersonalityEstimator(PersonalityEstimator):

    def __init__(self):
        self.openness_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.01
        )
        self.conscientiousness_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.01
        )
        self.extroversion_regressor = GradientBoostingRegressor(
            n_estimators=24,
            max_depth=3,
            learning_rate=0.01
        )
        self.agreeableness_regressor = GradientBoostingRegressor(
            n_estimators=24,
            max_depth=3,
            learning_rate=0.01
        )
        self.neuroticism_regressor = GradientBoostingRegressor(
            n_estimators=24,
            max_depth=3,
            learning_rate=0.01
        )

# get targets that are related to a personality trait for each user in liwc or nrc
    def _extract_targets(self, df: pd.DataFrame, labels: List[UserLabels], personality_trait) -> np.array:
        user_id_to_label = {label.user_id: label.personality_traits.__getattribute__(personality_trait) for label in labels}
        targets = list()
        for _, row in df.iterrows():
            targets.append(user_id_to_label[row["userId"]])
        return targets

# get the features related to a user in liwc or nrc
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

        personality_regressor = {
            "openness": self.openness_regressor,
            "conscientiousness": self.conscientiousness_regressor,
            "extroversion": self.extroversion_regressor,
            "agreeableness": self.agreeableness_regressor,
            "neuroticism": self.neuroticism_regressor
        }

        for personality, regressor in personality_regressor.items():
            liwc_nrc_targets = self._extract_targets(liwc_nrc_df, labels, personality)
            X, y = shuffle(liwc_nrc_features.values, liwc_nrc_targets)
            regressor.fit(X, y)

    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame) -> List[PersonalityTraits]:

        ope_predictions,con_predictions,ext_predictions,agr_predictions,nev_predictions = (list() for _ in range(5))

        liwc_nrc_df = self._merge_features(liwc_df, nrc_df)

        for feature in features:
            liwc_nrc_feature = self._extract_features(liwc_nrc_df, feature.user_id) # get the features for one specific user
            if liwc_nrc_feature is None:
                ope_predictions.append(float(PersonalityEstimator.OPENNESS_BASLINE))
                con_predictions.append(float(PersonalityEstimator.CONSCIENTIOUS_BASELINE))
                ext_predictions.append(float(PersonalityEstimator.EXTROVERT_BASELINE))
                agr_predictions.append(float(PersonalityEstimator.AGREEABLE_BASELINE))
                nev_predictions.append(float(PersonalityEstimator.NEUROTIC_BASELINE))
            else:
                liwc_nrc_feature = liwc_nrc_feature.drop(["userId"], axis=1)
                ope_predictions.append(float(self.openness_regressor.predict(liwc_nrc_feature)))
                con_predictions.append(float(self.conscientiousness_regressor.predict(liwc_nrc_feature)))
                ext_predictions.append(float(self.extroversion_regressor.predict(liwc_nrc_feature)))
                agr_predictions.append(float(self.agreeableness_regressor.predict(liwc_nrc_feature)))
                nev_predictions.append(float(self.neuroticism_regressor.predict(liwc_nrc_feature)))

        return [
            PersonalityTraits(
                openness=ope_predictions[i],
                conscientiousness=con_predictions[i],
                extroversion=ext_predictions[i],
                agreeableness=agr_predictions[i],
                neuroticism=nev_predictions[i]
            )
            for i in range(len(features))
        ]
