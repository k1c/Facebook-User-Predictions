from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.personality_estimator import PersonalityEstimator
from evaluation.evaluation_utils import regression_score

from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle


class GBDTPersonalityEstimator(PersonalityEstimator):

    def __init__(self, valid_split: float):
        self.valid_split = valid_split
        self.model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3
        )

    # def _extract_targets(self, oxford_df: pd.DataFrame, labels: List[UserLabels]) -> np.array:
    #     user_id_to_gender = {label.user_id: label.gender for label in labels}
    #     targets = list()
    #     for _, row in oxford_df.iterrows():
    #         targets.append(user_id_to_gender[row["userId"]])
    #     return targets
    #
    # def _extract_oxford_features(self, oxford_df: pd.DataFrame, user_id: str) -> Optional[pd.DataFrame]:
    #     rows = oxford_df[oxford_df["userId"] == user_id]
    #     if len(rows) == 0:
    #         return None
    #     elif len(rows) == 1:
    #
    #         return rows.to_frame().T if isinstance(rows, pd.Series) else rows
    #     else:
    #         row = rows.iloc[0]
    #         return row.to_frame().T if isinstance(row, pd.Series) else row

    def fit(self,
            features: List[UserFeatures],
            liwc_df: pd.DataFrame,
            nrc_df: pd.DataFrame,
            labels: List[UserLabels]) -> None:
        train_features, train_labels, valid_features, valid_labels = self.train_valid_split(
            features,
            labels,
            valid_split=self.valid_split
        )

        valid_predictions = self.predict(valid_features, liwc_df, nrc_df)
        scores = regression_score(predicted=valid_predictions, true=[x.personality_traits for x in valid_labels])
        print(scores)



        # targets = self._extract_targets(oxford_df, labels)
        # smote = SMOTE()
        # X, y = shuffle(features.values, targets)
        # X_res, y_res = smote.fit_resample(X, y)
        self.model.fit(X_res, y_res)

    def _predict_for_user(self, feature: UserFeatures) -> PersonalityTraits:

        # similar_users = defaultdict(int)
        #
        # for user_like in feature.likes:
        #     users_that_like_this = self.like_graph.get(user_like, list())
        #     for user in users_that_like_this:
        #         similar_users[user] += 1
        #
        # similar_users_sorted = sorted(
        #     list(similar_users.keys()),
        #     key=lambda x: similar_users.get(x),
        #     reverse=True
        # )
        #
        # num_similar_users = self.num_similar_users if len(similar_users) > self.num_similar_users \
        #     else len(similar_users)
        #
        # top_similar_users = set(similar_users_sorted[:num_similar_users])
        #
        # top_similar_users_count = self._normalize_counts(
        #     {user: similar_users[user] for user in similar_users if user in top_similar_users}
        # )

        openness, conscientiousness, extroversion, agreeableness, neuroticism = 0.0, 0.0, 0.0, 0.0, 0.0

        for user in top_similar_users:
            user_labels = self.user_label_graph.get(user)
            normalization = top_similar_users_count.get(user)
            openness += normalization * user_labels.personality_traits.openness
            conscientiousness += normalization * user_labels.personality_traits.conscientiousness
            extroversion += normalization * user_labels.personality_traits.extroversion
            agreeableness += normalization * user_labels.personality_traits.agreeableness
            neuroticism += normalization * user_labels.personality_traits.neuroticism

        return PersonalityTraits(
            openness=openness,
            conscientiousness=conscientiousness,
            extroversion=extroversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism
        )

    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame) -> List[PersonalityTraits]:
        return [self._predict_for_user(feature) for feature in features]
