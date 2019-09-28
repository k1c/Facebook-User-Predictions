#!/usr/bin/env python3

import argparse

from data.fb_user_labels import FBUserLabels
from data.readers import read_prediction_data
from estimators.fb_user_estimator import FBUserEstimator


def main(arguments: argparse.Namespace):
    fb_user_estimator = FBUserEstimator.load(arguments.model_path)
    features = read_prediction_data(arguments.input_path)

    predictions = fb_user_estimator.predict(features)
    FBUserLabels.save(
        predictions=predictions,
        save_path=arguments.output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("-m", "--model_path", type=str)
    args = parser.parse_args()
    main(args)
