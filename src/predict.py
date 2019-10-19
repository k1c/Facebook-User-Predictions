#!/usr/bin/env python3

import argparse
import os

from data.user_labels import UserLabels
from data.readers import read_prediction_data, read_liwc, read_nrc, read_oxford
from estimators.fb_user_estimator import FBUserEstimator
from util.utils import get_current_timestamp


def main(arguments: argparse.Namespace):
    print("")
    print("Loading model from '{}' ...".format(arguments.model_path))
    fb_user_estimator = FBUserEstimator.load(arguments.model_path)
    print("Loading test data from '{}' ...".format(arguments.input_path))
    features = read_prediction_data(arguments.input_path)
    liwc_df, nrc_df = read_liwc(arguments.data_path), read_nrc(arguments.data_path)
    oxford_df = read_oxford(arguments.data_path)
    print("Predicting labels for test data ...")
    predictions = fb_user_estimator.predict(
        features,
        liwc_df=liwc_df,
        nrc_df=nrc_df,
        oxford_df=oxford_df
    )

    save_path = UserLabels.save(
        predictions=predictions,
        save_path=arguments.output_path
    )
    print("Done! Predictions saved to '{}'".format(os.path.join(save_path, '*.xml')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_output_path = "predictions/{}/".format(get_current_timestamp())
    parser.add_argument("-i", "--input_path", type=str, default="../new_data/Public_Test/")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path)
    parser.add_argument("-m", "--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
