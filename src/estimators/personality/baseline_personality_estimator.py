from typing import List

import numpy as np
import pandas as pd

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.personality_estimator import PersonalityEstimator


class BaselinePersonalityEstimator(PersonalityEstimator):
    def __init__(self):
        self.predictions: np.array = None
        self.valid_split = 0.8

    def fit(self,
            features: List[UserFeatures],
            liwc_df: pd.DataFrame,
            nrc_df: pd.DataFrame,
            labels: List[UserLabels]) -> None:

        personalities: List[List[float]] = [label.personality_traits.as_list() for label in labels]

        self.predictions = np.mean(
            np.array(personalities),
            axis=0   # Take means of all columns
        )

    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame) -> List[PersonalityTraits]:

        return [
            PersonalityTraits(
                openness=self.predictions[0],
                conscientiousness=self.predictions[1],
                extroversion=self.predictions[2],
                agreeableness=self.predictions[3],
                neuroticism=self.predictions[4]
            )
            for _ in range(len(features))
        ]
