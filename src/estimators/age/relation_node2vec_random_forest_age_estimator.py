from typing import List

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.age_estimator import AgeEstimator
from data.readers import age_category_to_int
from data.pre_processors import get_node2vec_embeddings
from estimators.random_search import RandomSearch
from data.readers import int_category_to_age


class RelationNode2VecRandomForestAgeEstimator(AgeEstimator):
    def __init__(self):
        self.best_node2vec_hyper_parameters = {}
        self.best_model_parameters = {}
        self.best_estimator = None

    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            train_size=0.8,
            shuffle=True,
            random_state=8
        )

        node2vec_hyper_params = [
            # Hyper-parameters 1
            {
                'dimensions': 16,
                'walk_length': 20,
                'num_walks': 10,
                'p': 1,
                'q': 1
            },
            # Hyper-parameters 2
            {
                'dimensions': 16,
                'walk_length': 20,
                'num_walks': 10,
                'p': 2,
                'q': 2
            },
            # Hyper-parameters 3
            {
                'dimensions': 16,
                'walk_length': 20,
                'num_walks': 10,
                'p': 0.5,
                'q': 0.5
            },
            # Hyper-parameters 4
            {
                'dimensions': 16,
                'walk_length': 20,
                'num_walks': 10,
                'p': 0.5,
                'q': 2
            },
            # Hyper-parameters 5
            {
                'dimensions': 16,
                'walk_length': 20,
                'num_walks': 10,
                'p': 2,
                'q': 0.5
            }
        ]

        best_model_parameters = []
        best_model_scores = []
        for hyper_params in node2vec_hyper_params:
            x_train = get_node2vec_embeddings(x_train, hyper_params)
            y_train = np.array([age_category_to_int(label.age) for label in y_train])

            x_test = get_node2vec_embeddings(x_test, hyper_params)
            y_test = np.array([age_category_to_int(label.age) for label in y_test])

            best_model_params, best_model_score = RandomSearch.random_forest(100, x_train, y_train, x_test, y_test)
            best_model_parameters.append(best_model_params)
            best_model_scores.append(best_model_score)

        best_index = int(np.argmax(best_model_scores))
        self.best_node2vec_hyper_parameters = node2vec_hyper_params[best_index]
        self.best_model_parameters = best_model_parameters[best_index]
        print("Best test set score was {}".format(best_model_scores[best_index]))
        print("Best hyper-parameters were {}".format(self.best_node2vec_hyper_parameters))
        print("Best model parameters were {}".format(self.best_model_parameters))

        self.best_estimator = RandomForestClassifier(**self.best_model_parameters)
        self.best_estimator.fit(x_train, y_train)

    def predict(self, features: List[UserFeatures]) -> List[str]:
        embeddings = get_node2vec_embeddings(features, self.best_node2vec_hyper_parameters)
        predictions = self.best_estimator.predict(embeddings)
        return [int_category_to_age(int(prediction)) for prediction in predictions]
