from typing import List

import numpy as np

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.personality_estimator import PersonalityEstimator


class BaselinePersonalityEstimator(PersonalityEstimator):
    def __init__(self):
        self.predictions: np.array = None

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        personalities: List[List[float]] = [label.personality_traits.as_list() for label in labels]

        self.predictions = np.mean(
            np.array(personalities),
            axis=0   # Take means of all columns
        )

    def predict(self, features: List[FBUserFeatures]) -> List[PersonalityTraits]:
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
