from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from estimators.base.gender_estimator import GenderEstimator


class CnnGenderEstimator(GenderEstimator):

    def __init__(self, neural_net):
        self.neural_net = neural_net

    def dataloader(self, features, labels):
        pass

    def fit(self, features, labels):
        pass

    def predict(self):
        pass