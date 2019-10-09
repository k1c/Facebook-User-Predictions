from collections import defaultdict
from typing import List, DefaultDict, Dict

from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.personality_estimator import PersonalityEstimator
from evaluation_utils import regression_score


class GraphSimilarityPersonalityEstimator(PersonalityEstimator):
    def __init__(self, num_similar_users: int, valid_split: float):
        self.valid_split = valid_split
        self.num_similar_users = num_similar_users
        self.like_graph: DefaultDict[int, List[str]] = None
        self.user_label_graph: Dict[str, FBUserLabels] = None

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]) -> None:
        train_features, train_labels, valid_features, valid_labels = self.train_valid_split(
            features,
            labels,
            valid_split=self.valid_split
        )

        self.like_graph: DefaultDict[int, List[str]] = self._build_like_graph(train_features)
        self.user_label_graph: Dict[str, FBUserLabels] = self._build_user_label_graph(train_labels)

        valid_predictions = self.predict(valid_features)
        scores = regression_score(predicted=valid_predictions, true=[x.personality_traits for x in valid_labels])
        print(scores)

    @staticmethod
    def _build_like_graph(features: List[FBUserFeatures]) -> DefaultDict[int, List[str]]:
        like_graph = defaultdict(list)
        for feature in features:
            for like in feature.likes:
                like_graph[like].append(feature.user_id)
        return like_graph

    @staticmethod
    def _build_user_label_graph(labels: List[FBUserLabels]) -> Dict[str, FBUserLabels]:
        return {
            label.user_id: label for label in labels
        }

    def _normalize_counts(self, similar_users: Dict[str, int]) -> Dict[str, float]:
        normalizing_constant = sum(similar_users.values())
        return {
            user: (float(similar_users.get(user)) / normalizing_constant) for user in similar_users.keys()
        }

    def _predict_for_user(self, feature: FBUserFeatures) -> PersonalityTraits:
        similar_users = defaultdict(int)
        for user_like in feature.likes:
            users_that_like_this = self.like_graph.get(user_like, list())
            for user in users_that_like_this:
                similar_users[user] += 1

        similar_users_sorted = sorted(
            list(similar_users.keys()),
            key=lambda x: similar_users.get(x),
            reverse=True
        )

        num_similar_users = self.num_similar_users if len(similar_users) > self.num_similar_users \
            else len(similar_users)

        top_similar_users = set(similar_users_sorted[:num_similar_users])
        print(top_similar_users)
        top_similar_users_count = self._normalize_counts(
            {user: similar_users[user] for user in similar_users if user in top_similar_users}
        )
        openness, conscientiousness, extroversion, agreeableness, neuroticism = 0.0, 0.0, 0.0, 0.0, 0.0

        for user in top_similar_users:
            user_labels = self.user_label_graph.get(user)
            normalization = top_similar_users_count.get(user)
            openness += normalization * user_labels.personality_traits.openness
            conscientiousness += normalization * user_labels.personality_traits.conscientiousness
            extroversion += normalization * user_labels.personality_traits.extroversion
            agreeableness += normalization * user_labels.personality_traits.agreeableness
            neuroticism += normalization * user_labels.personality_traits.neuroticism

        return PersonalityTraits(
            openness=openness,
            conscientiousness=conscientiousness,
            extroversion=extroversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism
        )

    def predict(self, features: List[FBUserFeatures]) -> List[PersonalityTraits]:
        return [self._predict_for_user(feature) for feature in features]
