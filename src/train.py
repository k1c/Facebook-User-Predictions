import argparse
import uuid

from data.readers import read_train_data
from estimators.age.baseline_age_estimator import BaselineAgeEstimator
from estimators.fb_user_estimator import FBUserEstimator
from estimators.gender.baseline_gender_estimator import BaselineGenderEstimator
from estimators.personality.baseline_personality_estimator import BaselinePersonalityEstimator

age_estimators = {
    'baseline': BaselineAgeEstimator
}

gender_estimators = {
    'baseline': BaselineGenderEstimator
}

personality_estimators = {
    'baseline': BaselinePersonalityEstimator
}


def main(arguments: argparse.Namespace):
    features, labels = read_train_data(data_path=arguments.data_path)

    fb_user_estimator = FBUserEstimator(
        model_id=uuid.uuid4().hex,
        age_estimator=age_estimators.get(arguments.age_estimator)(),
        gender_estimator=gender_estimators.get(arguments.gender_estimator)(),
        personality_estimator=personality_estimators.get(arguments.personality_estimator)()
    )

    fb_user_estimator.fit(features, labels)
    fb_user_estimator.save(arguments.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--age_estimator", type=str, choices=age_estimators.keys())
    parser.add_argument("--gender_estimator", type=str, choices=gender_estimators.keys())
    parser.add_argument("--personality_estimator", choices=personality_estimators.keys())
    parser.add_argument("--save_path", type=str, default=".")
    args = parser.parse_args()
    main(args)
