import os
import pickle
from typing import List

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.age_estimator import AgeEstimator
from estimators.base.gender_estimator import GenderEstimator
from estimators.base.personality_estimator import PersonalityEstimator


class FBUserEstimator:

    def __init__(self,
                 model_id: str,
                 age_estimator: AgeEstimator,
                 gender_estimator: GenderEstimator,
                 personality_estimator: PersonalityEstimator):

        self.model_id = model_id
        self.age_estimator = age_estimator
        self.gender_estimator = gender_estimator
        self.personality_estimator = personality_estimator

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]):
        self.age_estimator.fit(features, labels)
        self.gender_estimator.fit(features, labels)
        self.personality_estimator.fit(features, labels)

    def predict(self, features: List[FBUserFeatures]) -> List[FBUserLabels]:

        age_predictions: List[str] = self.age_estimator.predict(features)
        gender_predictions: List[int] = self.gender_estimator.predict(features)
        personality_predictions: List[PersonalityTraits] = self.personality_estimator.predict(features)

        return [
            FBUserLabels.from_data(
                user_id=features[idx].user_id,
                age=age_predictions[idx],
                gender=gender_predictions[idx],
                personality_traits=personality_predictions[idx]
            )
            for idx in range(len(features))
        ]

    # Model persistence
    def save(self, save_path: str):
        file_name = os.path.join(
            save_path,
            self.model_id
        )
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(load_path: str, model_id: str) -> 'FBUserEstimator':
        file_name = os.path.join(
            load_path,
            model_id
        )
        with open(file_name, "rb") as f:
            model = pickle.load(f)

        return model
