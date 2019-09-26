from abc import abstractmethod, ABC
from typing import List

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels


class GenderEstimator(ABC):

    @abstractmethod
    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        pass

    @abstractmethod
    def predict(self, features: List[FBUserFeatures]) -> List[int]:
        pass
