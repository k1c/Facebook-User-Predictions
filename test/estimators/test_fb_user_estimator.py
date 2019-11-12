import unittest
import pandas as pd

from data.readers import read_train_data
from estimators.age.baseline_age_estimator import BaselineAgeEstimator
from estimators.fb_user_estimator import FBUserEstimator
from estimators.gender.baseline_gender_estimator import BaselineGenderEstimator
from estimators.personality.baseline_personality_estimator import BaselinePersonalityEstimator


class TestFBUserEstimator(unittest.TestCase):

    def setUp(self):
        self.features, self.labels = read_train_data("../../datasets/synthetic")
        self.liwc_df = pd.DataFrame()
        self.nrc_df = pd.DataFrame()
        self.oxford_df = pd.DataFrame()

    def test_prediction(self):
        fb_user_estimator = FBUserEstimator(
            model_id="foobar",
            age_estimator=BaselineAgeEstimator(),
            gender_estimator=BaselineGenderEstimator(),
            personality_estimator=BaselinePersonalityEstimator()
        )
        fb_user_estimator.fit(
            self.features,
            self.liwc_df,
            self.nrc_df,
            self.oxford_df,
            self.labels
        )
