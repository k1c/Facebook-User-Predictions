import os
from collections import defaultdict
from typing import List, Tuple, DefaultDict

import pandas as pd

from constants.column_names import USER_ID, AGE, GENDER, OPENNESS, CONSCIENTIOUSNESS, EXTROVERSION, AGREEABLENESS, \
    NEUROTICISM, LIKE_ID
from constants.directory_names import PROFILE_DIR, LIKES_DIR, RELATION_DIR
from constants.file_names import PROFILE_FILE, RELATIONS_FILE
from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from data.personality_traits import PersonalityTraits


def read_statuses(data_path: str, user_id: str) -> List[str]:
    """
    Reads the status message of the user in the `Text` sub folder. Each user has a text file with status messages
    titled `Text/user_id.txt`.
    :param data_path: Path to the main directory that contains the `Text` directory.
    :param user_id: The hex id of the user
    :return:
    """
    status_file_path = os.path.join(
        data_path,
        LIKES_DIR,
        "{}.txt".format(user_id)
    )

    with open(status_file_path) as f:
        content = f.readlines()

    return [
        line.strip() for line in content
    ]


def read_image(data_path: str, user_id: str):
    return 1


def read_likes(data_path: str) -> DefaultDict[str, List[int]]:
    """
    Reads the like ids of the user in the `Relation` sub-folder in the `Relation.csv` file.
    The file has three columns, `idx`, `userid`, `like_id` and is directly read into a pandas DataFrame.
    :param data_path:
    :return:
    """
    likes_file_path = os.path.join(
        data_path,
        RELATION_DIR,
        RELATIONS_FILE
    )

    like_df = pd.read_csv(likes_file_path)
    user_likes: DefaultDict[str, List[int]] = defaultdict(list)
    for _, row in like_df.iterrows():
        user_likes[row[USER_ID]].append(row[LIKE_ID])
    return user_likes


def categorize_age(age: str) -> str:
    age = float(age)
    if 0 <= age <= 24:
        return "xx-24"
    elif 25 <= age <= 34:
        return "25-34"
    elif 35 <= age <= 49:
        return "35-49"
    else:
        return "50-xx"


def read_train_data(data_path: str) -> Tuple[List[FBUserFeatures], List[FBUserLabels]]:
    """
    The main entry point to read all the data. It goes through the `Profile/Profile.csv` file, gets the `userid`,
    and collects the associated status messages, likes and image for the user and encapsulates them in the
    `FBUserFeatures` and `FBUserLabels` objects.
    :param data_path: The path of the directory that contains the `Text`, `Profile`, `Relation` and `Image` sub-folders.
    :return:
    """
    profile_file_path = os.path.join(
        data_path,
        PROFILE_DIR,
        PROFILE_FILE
    )

    profile_df = pd.read_csv(
        profile_file_path,
        converters={
            AGE: categorize_age
        }
    )
    likes = read_likes(data_path)
    features, labels = list(), list()

    for _, row in profile_df.iterrows():
        user_id = row[USER_ID]

        features.append(
            FBUserFeatures.from_data(
                user_id=user_id,
                likes=likes[user_id],
                statuses=read_statuses(data_path, user_id),
                image=read_image(data_path, user_id)
            )
        )

        labels.append(
            FBUserLabels.from_data(
                user_id=user_id,
                age=row[AGE],
                gender=row[GENDER],
                personality_traits=PersonalityTraits(
                    openness=row[OPENNESS],
                    conscientiousness=row[CONSCIENTIOUSNESS],
                    extroversion=row[EXTROVERSION],
                    agreeableness=row[AGREEABLENESS],
                    neuroticism=row[NEUROTICISM]
                )
            )
        )

    return features, labels


def read_prediction_data(data_path: str) -> List[FBUserFeatures]:
    profile_file_path = os.path.join(
        data_path,
        PROFILE_DIR,
        PROFILE_FILE
    )

    profile_df = pd.read_csv(
        profile_file_path,
        converters={
            AGE: categorize_age
        }
    )
    likes = read_likes(data_path)
    features, labels = list(), list()

    for _, row in profile_df.iterrows():
        user_id = row[USER_ID]

        features.append(
            FBUserFeatures.from_data(
                user_id=user_id,
                likes=likes[user_id],
                statuses=read_statuses(data_path, user_id),
                image=read_image(data_path, user_id)
            )
        )
    return features

