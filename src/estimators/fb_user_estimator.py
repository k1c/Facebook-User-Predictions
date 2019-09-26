from typing import List

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.age_estimator import AgeEstimator
from estimators.base.gender_estimator import GenderEstimator
from estimators.base.personality_estimator import PersonalityEstimator


class FBUserEstimator:

    def __init__(self,
                 age_estimator: AgeEstimator,
                 gender_estimator: GenderEstimator,
                 personality_estimator: PersonalityEstimator):

        self.age_estimator = age_estimator
        self.gender_estimator = gender_estimator
        self.personality_estimator = personality_estimator

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]):
        self.age_estimator.fit(features, labels)
        self.gender_estimator.fit(features, labels)
        self.personality_estimator.fit(features, labels)

    def predict(self, features: List[FBUserFeatures]) -> List[FBUserLabels]:

        ages: List[str] = self.age_estimator.predict(features)
        genders: List[int] = self.gender_estimator.predict(features)
        personality_traits: List[PersonalityTraits] = self.personality_estimator.predict(features)

        return [
            FBUserLabels.from_predictions(
                user_id=features[idx].user_id,
                age=ages[idx],
                gender=genders[idx],
                personality_traits=personality_traits[idx]
            )
            for idx in range(len(features))
        ]

    # Model persistence
    def save(self):
        pass

    def load(self):
        pass
