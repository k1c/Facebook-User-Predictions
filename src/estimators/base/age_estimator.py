from abc import abstractmethod, ABC

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from typing import List


class AgeEstimator(ABC):

    @abstractmethod
    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        pass

    @abstractmethod
    def predict(self, features: List[FBUserFeatures]) -> List[str]:
        pass
