import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


def remove_outliers(oxford_df: pd.DataFrame):

    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=100, random_state=rng, contamination=.1)

    oxford_new_df = oxford_df.copy()
    x_train = oxford_new_df.values[:, 2:]

    clf.fit(x_train)
    y_pred_train = clf.predict(x_train)

    oxford_new_df = oxford_new_df[np.where(y_pred_train == 1, True, False)]

    return oxford_new_df


def create_face_area(oxford_df: pd.DataFrame):

    oxford_new_df = oxford_df.copy()
    oxford_new_df = oxford_new_df.assign(faceRectangle_area= \
                                         lambda x: oxford_new_df['faceRectangle_width'] * \
                                         oxford_new_df['faceRectangle_height'])

    return oxford_new_df


def remove_duplicates(oxford_df: pd.DataFrame):

    oxford_new_df = oxford_df.copy()
    duplicates = oxford_new_df[oxford_new_df.duplicated(['userId'])]

    for userid in duplicates['userId']:
        df = oxford_new_df[oxford_new_df['userId'] == userid]
        faceid_used = df[df['faceRectangle_area'] == df['faceRectangle_area'].max()]["faceID"]
        oxford_new_df = oxford_new_df[oxford_new_df['faceID'] != faceid_used.values[0]]

    return oxford_new_df
