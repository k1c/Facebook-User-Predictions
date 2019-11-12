from abc import abstractmethod
from typing import List

import pandas as pd

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.base_estimator import BaseEstimator


class PersonalityEstimator(BaseEstimator):

    @abstractmethod
    def fit(self,
            features: List[UserFeatures],
            liwc_df: pd.DataFrame,
            nrc_df: pd.DataFrame,
            labels: List[UserLabels]) -> None:
        pass

    @abstractmethod
    def predict(self,
                features: List[UserFeatures],
                liwc_df: pd.DataFrame,
                nrc_df: pd.DataFrame) -> List[PersonalityTraits]:
        pass
