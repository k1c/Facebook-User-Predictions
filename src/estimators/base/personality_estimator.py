from abc import abstractmethod
from typing import List

import pandas as pd

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.base_estimator import BaseEstimator


class PersonalityEstimator(BaseEstimator):

    EXTROVERT_BASELINE = 3.486857894736844
    NEUROTIC_BASELINE = 2.732424210526319
    AGREEABLE_BASELINE = 3.5839042105263132
    CONSCIENTIOUS_BASELINE = 3.44561684210526
    OPENNESS_BASLINE = 3.908690526315789

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
