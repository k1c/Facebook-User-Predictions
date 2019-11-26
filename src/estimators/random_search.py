import copy
import random
from sklearn.ensemble import RandomForestClassifier


class RandomSearch:
    @staticmethod
    def random_forest(
        num_searches,
        x_train,
        y_train,
        x_test,
        y_test
    ):
        best_model_params = {}
        best_test_score = 0.0
        for i in range(num_searches):
            model_params = {
                'n_estimators': random.randint(1, 100),
                'criterion': random.choice(['gini', 'entropy']),
                'max_depth': random.randint(30, 100),
                'min_samples_split': random.randint(2, 40),
                'min_samples_leaf': random.randint(1, 20),
                'min_weight_fraction_leaf': 0.5 * random.random(),
                'max_features': random.random(),
                'max_leaf_nodes': random.randint(2, 100),
                'min_impurity_decrease': random.random(),
                'bootstrap': random.choice([False, True])
            }
            model_params['oob_score'] = random.choice([False, True]) if model_params['bootstrap'] else False

            estimator = RandomForestClassifier(**model_params)

            estimator.fit(x_train, y_train)
            test_score = estimator.score(x_test, y_test)

            if test_score > best_test_score:
                best_test_score = test_score
                best_model_params = copy.deepcopy(model_params)

        return best_model_params, best_test_score
