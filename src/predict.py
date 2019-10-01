#!/usr/bin/env python3

import argparse
import os

from data.fb_user_labels import FBUserLabels
from data.readers import read_prediction_data
from estimators.fb_user_estimator import FBUserEstimator
from utils import get_current_timestamp


def main(arguments: argparse.Namespace):
    print("Loading model from '{}' ...".format(arguments.model_path))
    fb_user_estimator = FBUserEstimator.load(arguments.model_path)
    print("Loading test data from '{}' ...".format(arguments.input_path))
    features = read_prediction_data(arguments.input_path)

    print("Predicting labels for test data ...")
    predictions = fb_user_estimator.predict(features)

    save_path = FBUserLabels.save(
        predictions=predictions,
        save_path=arguments.output_path
    )
    print("Done! Predictions saved to '{}'".format(os.path.join(save_path, '*.xml')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_output_path = "predictions/{}/".format(get_current_timestamp())
    parser.add_argument("-i", "--input_path", type=str, default="../data/Public_Test/")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path)
    parser.add_argument("-m", "--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
