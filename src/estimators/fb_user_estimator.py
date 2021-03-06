import os
import pickle
from typing import List
import pathlib
import pandas as pd

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.age_estimator import AgeEstimator
from estimators.base.gender_estimator import GenderEstimator
from estimators.base.personality_estimator import PersonalityEstimator
from util.utils import get_current_timestamp


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

    def fit(self,
            features: List[UserFeatures],
            liwc_df: pd.DataFrame,
            nrc_df: pd.DataFrame,
            oxford_df: pd.DataFrame,
            labels: List[UserLabels]):
        self.personality_estimator.fit(features, liwc_df, nrc_df, labels)
        self.age_estimator.fit(features, liwc_df, nrc_df, labels)
        self.gender_estimator.fit(features, oxford_df, labels)

    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame,
                oxford_df: pd.DataFrame) -> List[UserLabels]:

        age_predictions: List[str] = self.age_estimator.predict(features,liwc_df, nrc_df)
        gender_predictions: List[int] = self.gender_estimator.predict(features, oxford_df)
        personality_predictions: List[PersonalityTraits] = self.personality_estimator.predict(features, liwc_df, nrc_df)
        return [
            UserLabels(
                user_id=features[idx].user_id,
                age=age_predictions[idx],
                gender=gender_predictions[idx],
                personality_traits=personality_predictions[idx]
            )
            for idx in range(len(features))
        ]

    # Model persistence
    def save(self, save_path: str) -> str:
        model_file_name = "{}_{}.pkl".format(self.model_id, get_current_timestamp())
        model_path = os.path.join(
            save_path,
            model_file_name
        )
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

        return model_path

    @staticmethod
    def load(model_path: str) -> 'FBUserEstimator':
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        return model
