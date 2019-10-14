from typing import List

import numpy as np

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.personality_estimator import PersonalityEstimator
from evaluation_utils import regression_score


class BaselinePersonalityEstimator(PersonalityEstimator):
    def __init__(self):
        self.predictions: np.array = None
        self.valid_split = 0.8

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        train_features, train_labels, valid_features, valid_labels = self.train_valid_split(
            features,
            labels,
            valid_split=self.valid_split
        )

        personalities: List[List[float]] = [label.personality_traits.as_list() for label in train_labels]

        self.predictions = np.mean(
            np.array(personalities),
            axis=0   # Take means of all columns
        )
        valid_predictions = self.predict(valid_features)
        scores = regression_score(predicted=valid_predictions, true=[x.personality_traits for x in valid_labels])
        print(scores)
        
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
