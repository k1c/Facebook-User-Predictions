import os
from collections import defaultdict
from typing import List, Tuple, DefaultDict

import pandas as pd
import numpy as np
from matplotlib import image as img

from constants.column_names import USER_ID, AGE, GENDER, OPENNESS, CONSCIENTIOUSNESS, EXTROVERSION, AGREEABLENESS, \
    NEUROTICISM, LIKE_ID
from constants.directory_names import PROFILE_DIR, LIKES_DIR, RELATION_DIR, IMAGE_DIR
from constants.file_names import PROFILE_FILE, RELATIONS_FILE
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.personality_traits import PersonalityTraits
from data.pre_processors import pre_process_likes_v1


def read_statuses_of_user(data_path: str, user_id: str) -> List[str]:
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

    with open(status_file_path, "r", errors="replace") as f:
        content = f.readlines()

    return [
        line.strip() for line in content
    ]


def read_image_of_user(data_path: str, user_id: str) -> np.ndarray:
    """
    Reads the image of the user in the `Image` sub folder. Each user has a jpg file with a profile picture
    titled `Image/user_id.jpg`.
    :param data_path: Path to the main directory that contains the `Image` directory.
    :param user_id: The hex id of the user
    :return: image: a numpy array
    """
    image_file_path = os.path.join(
        data_path,
        IMAGE_DIR,
        "{}.jpg".format(user_id)
    )

    image = img.imread(image_file_path)
    return image


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


def age_category_to_int(age_category: str) -> int:
    if age_category == "xx-24":
        return 0
    elif age_category == "25-34":
        return 1
    elif age_category == "35-49":
        return 2
    else:
        return 3


def int_to_age_category(int_age_category: int) -> str:
    if int_age_category == 0:
        return "xx-24"
    elif int_age_category == 1:
        return "25-34"
    elif int_age_category == 2:
        return "35-49"
    else:
        return "50-xx"


def read_train_data(data_path: str) -> Tuple[List[UserFeatures], List[UserLabels]]:
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
    ).sort_values(  # So that custom preprocessed data can be in the same order
        by=['userid']
    )
    likes = read_likes(data_path)
    features, labels = list(), list()

    # Pre-processing
    likes_preprocessed_v1 = pre_process_likes_v1(data_path)

    for _, row in profile_df.iterrows():
        user_id = row[USER_ID]

        features.append(
            UserFeatures.from_data(
                user_id=user_id,
                likes=likes[user_id],
                likes_preprocessed_v1=likes_preprocessed_v1[
                    likes_preprocessed_v1['userid'] == user_id
                ][likes_preprocessed_v1.columns[1:]].to_numpy(),
                # TODO: Implement. Data structure has changed. Old code: read_statuses_of_user(data_path, user_id),
                statuses=[],
                # TODO: Implement. Data structure has changed. Old code: read_image_of_user(data_path, user_id)
                image=np.array([])
            )
        )

        labels.append(
            UserLabels.from_data(
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


def read_prediction_data(data_path: str) -> List[UserFeatures]:
    profile_file_path = os.path.join(
        data_path,
        PROFILE_DIR,
        PROFILE_FILE
    )

    profile_df = pd.read_csv(profile_file_path)
    likes = read_likes(data_path)
    features, labels = list(), list()

    # Pre-processing
    likes_preprocessed_v1 = pre_process_likes_v1(data_path)

    for _, row in profile_df.iterrows():
        user_id = row[USER_ID]

        features.append(
            UserFeatures.from_data(
                user_id=user_id,
                likes=likes[user_id],
                likes_preprocessed_v1=likes_preprocessed_v1[
                    likes_preprocessed_v1['userid'] == user_id
                ][likes_preprocessed_v1.columns[1:]].to_numpy(),
                statuses=read_statuses_of_user(data_path, user_id),
                image=read_image_of_user(data_path, user_id)
            )
        )
    return features
