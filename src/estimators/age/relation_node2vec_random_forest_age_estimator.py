from typing import List

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.age_estimator import AgeEstimator
from constants.hyper_parameters import RELATION_AGE_NODE2VEC_HYPER_PARAMS
from data.pre_processors import get_node2vec_embeddings
from estimators.estimator_utils import RandomSearch
from estimators.estimator_utils import get_age_embeddings_dataset_splits
from data.readers import int_category_to_age


class RelationNode2VecRandomForestAgeEstimator(AgeEstimator):
    def __init__(self):
        self.best_node2vec_hyper_parameters = {}
        self.best_model_parameters = {}
        self.best_estimator = None

    def fit(
        self,
        features: List[UserFeatures],
        liwc_df: pd.DataFrame,
        nrc_df: pd.DataFrame,
        labels: List[UserLabels]
    ) -> None:
        best_model_parameters = []
        best_model_scores = []
        for hyper_params in RELATION_AGE_NODE2VEC_HYPER_PARAMS:
            x_train, x_test, y_train, y_test = get_age_embeddings_dataset_splits(
                features,
                labels,
                hyper_params,
                get_node2vec_embeddings
            )
            best_model_params, best_model_score = RandomSearch.random_forest(100, x_train, y_train, x_test, y_test)
            best_model_parameters.append(best_model_params)
            best_model_scores.append(best_model_score)

        best_index = int(np.argmax(best_model_scores))
        self.best_node2vec_hyper_parameters = RELATION_AGE_NODE2VEC_HYPER_PARAMS[best_index]
        self.best_model_parameters = best_model_parameters[best_index]
        print("Best test set score was {}".format(best_model_scores[best_index]))
        print("Best hyper-parameters were {}".format(self.best_node2vec_hyper_parameters))
        print("Best model parameters were {}".format(self.best_model_parameters))

        self.best_estimator = RandomForestClassifier(**self.best_model_parameters)
        x_train, x_test, y_train, y_test = get_age_embeddings_dataset_splits(
            features,
            labels,
            self.best_node2vec_hyper_parameters,
            get_node2vec_embeddings
        )
        self.best_estimator.fit(x_train, y_train)

    def predict(
        self,
        features: List[UserFeatures],
        liwc_df: pd.DataFrame,
        nrc_df: pd.DataFrame
    ) -> List[str]:
        embeddings = get_node2vec_embeddings(features, self.best_node2vec_hyper_parameters)
        predictions = self.best_estimator.predict(embeddings)
        return [int_category_to_age(int(prediction)) for prediction in predictions]
