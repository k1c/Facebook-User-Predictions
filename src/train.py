import argparse

from data.readers import read_train_data, read_liwc, read_nrc, read_oxford
from estimators.age.baseline_age_estimator import BaselineAgeEstimator
from estimators.age.relation_v1_age_estimator import RelationV1AgeEstimator
from estimators.gender.relation_v1_gender_estimator import RelationV1GenderEstimator
from estimators.personality.relation_v1_personality_estimator import RelationV1PersonalityEstimator
from estimators.age.relation_deep_walk_age_estimator import RelationDeepWalkAgeEstimator
from estimators.fb_user_estimator import FBUserEstimator
from estimators.gender.baseline_gender_estimator import BaselineGenderEstimator
from estimators.gender.gbdt_gender_estimator import GBDTGenderEstimator
from estimators.gender.cnn_gender_estimator import CnnGenderEstimator
from estimators.personality.baseline_personality_estimator import BaselinePersonalityEstimator
from estimators.personality.graph_similarity_personality_estimator import GraphSimilarityPersonalityEstimator
from estimators.personality.bert_regression_personality_estimator import BertRegressionPersonalityEstimator
from util.utils import get_random_id


age_estimators = {
    'baseline': BaselineAgeEstimator,
    'relation_v1': RelationV1AgeEstimator,
    'relation_deep_walk': RelationDeepWalkAgeEstimator
}

gender_estimators = {
    'baseline': BaselineGenderEstimator,
    'cnn': CnnGenderEstimator,
    'relation_v1': RelationV1GenderEstimator,
    'gbdt': GBDTGenderEstimator
}

personality_estimators = {
    'baseline': BaselinePersonalityEstimator,
    'bert_regression': BertRegressionPersonalityEstimator,
    'graph_regression': GraphSimilarityPersonalityEstimator,
    'relation_v1': RelationV1PersonalityEstimator,
    'graph': GraphSimilarityPersonalityEstimator
}


def main(arguments: argparse.Namespace):
    print("\nLoading training data from '{}' ...".format(arguments.data_path))
    features, labels = read_train_data(data_path=arguments.data_path)

    print("Initialising estimators ...")
    fb_user_estimator = FBUserEstimator(
        model_id=get_random_id(),
        age_estimator=age_estimators.get(arguments.age_estimator)(),
        gender_estimator=gender_estimators.get(arguments.gender_estimator)(),
        personality_estimator=personality_estimators.get(arguments.personality_estimator)(valid_split=0.8)
    )
    liwc_df, nrc_df = read_liwc(arguments.data_path), read_nrc(arguments.data_path)
    oxford_df = read_oxford(arguments.data_path)
    print("Fitting estimators on training data ...")
    fb_user_estimator.fit(features, liwc_df, nrc_df, oxford_df, labels)

    model_path = fb_user_estimator.save(arguments.save_path)
    print("Done! Model saved. Set MODEL_PATH in submission/ift6758.py to '{}' "
          "if you want to set it as the default model for predictions.".format(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../new_data/Train/")
    parser.add_argument("--age_estimator", type=str, choices=age_estimators.keys(), required=True)
    parser.add_argument("--gender_estimator", type=str, choices=gender_estimators.keys(), required=True)
    parser.add_argument("--personality_estimator", choices=personality_estimators.keys(), required=True)
    parser.add_argument("--save_path", type=str, default="models/")
    args = parser.parse_args()
    main(args)
