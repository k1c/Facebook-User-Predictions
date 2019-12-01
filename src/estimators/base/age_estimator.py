from abc import abstractmethod
from typing import List
import pandas as pd

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.base_estimator import BaseEstimator


class AgeEstimator(BaseEstimator):

    AGE_BASELINE = 'xx-24'

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
                nrc_df: pd.DataFrame) -> List[str]:
        pass
