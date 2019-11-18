from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.personality_estimator import PersonalityEstimator
from evaluation.evaluation_utils import regression_score

from sklearn.utils import shuffle


class GBDTPersonalityEstimator(PersonalityEstimator):

    def __init__(self):
        self.valid_split = 0.8
        self.openness_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3
        )
        self.conscientiousness_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3
        )
        self.extraversion_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3
        )
        self.agreeableness_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3
        )
        self.neuroticism_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3
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

    def fit(self,
            features: List[UserFeatures],
            liwc_df: pd.DataFrame,
            nrc_df: pd.DataFrame,
            labels: List[UserLabels]) -> None:

        # train_features, train_labels, valid_features, valid_labels = self.train_valid_split(
        #     features,
        #     labels,
        #     valid_split=self.valid_split
        # )
        # Sometimes we have two rows with the same userId. There might be two different people of different
        # genders in this, and this will only confuse our algorithm. So we drop duplicates.
        # liwc_df = liwc_df.drop_duplicates(
        #     subset=["userId"],
        #     keep="first"
        # )
        features = liwc_df.drop(
            ["userId"],
            axis=1
        )
        personalities = ["openness", "conscientiousness","extroversion", "agreeableness", "neuroticism"]
        regressors = [self.openness_regressor, self.conscientiousness_regressor, self.extraversion_regressor, self.agreeableness_regressor, self.neuroticism_regressor]

        for personality, regressor in zip(personalities, regressors):
            liwc_targets = self._extract_targets(liwc_df, labels, personality)
            X, y = shuffle(features.values, liwc_targets)
            regressor.fit(X, y)

    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame) -> List[PersonalityTraits]:

        ope_predictions,con_predictions,ext_predictions,agr_predictions,nev_predictions = (list() for _ in range(5))

        for feature in features:
            liwc_feature = self._extract_features(liwc_df, feature.user_id)
            liwc_feature = liwc_feature.drop(["userId"], axis=1)
            ope_predictions.append(float(self.openness_regressor.predict(liwc_feature)))
            con_predictions.append(float(self.conscientiousness_regressor.predict(liwc_feature)))
            ext_predictions.append(float(self.extraversion_regressor.predict(liwc_feature)))
            agr_predictions.append(float(self.agreeableness_regressor.predict(liwc_feature)))
            nev_predictions.append(float(self.neuroticism_regressor.predict(liwc_feature)))

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
