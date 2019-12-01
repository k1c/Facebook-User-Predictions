import os.path
from data.user_features import UserFeatures
from typing import List
import subprocess
import os
import numpy as np
from util.utils import get_random_id
import itertools
from collections import Counter
from scipy import stats
from constants.directory_names import TEMP_DIR


def pre_process_likes_v1(user_features: List[UserFeatures]) -> np.array:
    def get_like_counts_per_user(_user_features: List[UserFeatures]):
        result = {}
        for user in _user_features:
            result[user.user_id] = len(user.likes)
        return result

    def get_like_counts_per_page(_user_features: List[UserFeatures]):
        all_likes_ids_lists = [user.likes for user in _user_features]
        all_likes_ids_lists_flat = [item for sublist in all_likes_ids_lists for item in sublist]
        return Counter(all_likes_ids_lists_flat)

    def get_pages_liked_sum_likes(_user: UserFeatures):
        return np.sum([
            like_counts_per_page[page_id]
            for page_id in _user.likes
        ])

    like_counts_per_user = get_like_counts_per_user(user_features)
    like_counts_per_page = get_like_counts_per_page(user_features)

    raw_result = np.array([
        np.array([
            like_counts_per_user[user.user_id],
            get_pages_liked_sum_likes(user)
        ])
        for user in user_features
    ])

    # Standardize features by removing the mean and scaling to unit variance
    return stats.zscore(raw_result)


def get_deep_walk_embeddings(features: List[UserFeatures]):
    input_edge_list_file = '{}/relations_edge_list_{}.txt'.format(TEMP_DIR, get_random_id())
    output_embeddings_file = '{}/relations_embeddings_{}.txt'.format(TEMP_DIR, get_random_id())
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    create_deep_walk_edge_list_file(input_edge_list_file, features)

    # This library assumes it is used as a process, not a library. Create embeddings:
    print("Running the DeepWalk algorithm to produce embeddings for user likes...")
    subprocess.Popen(
        ' '.join([
            'deepwalk',
            '--input {}'.format(input_edge_list_file),
            '--output {}'.format(output_embeddings_file),
            '--format edgelist',
            '--workers 4'
        ]),
        shell=True
    ).wait()

    embeddings = np.genfromtxt(output_embeddings_file, delimiter=' ', skip_header=1)
    embeddings_sorted = embeddings[embeddings[:, 0].argsort()]
    os.remove(input_edge_list_file)
    os.remove(output_embeddings_file)
    return embeddings_sorted[:9500, 1:]


def create_deep_walk_edge_list_file(file_name: str, features: List[UserFeatures]):
    all_user_ids = [feature.user_id for feature in features]
    all_likes_ids = list(itertools.chain(*[feature.likes for feature in features]))
    all_unique_ids = all_user_ids + list(set(all_likes_ids))
    index_dict = {
        id_: index
        for index, id_ in enumerate(all_unique_ids)
    }
    with open(file_name, 'w') as file_handler:
        for feature in features:
            for like_id in feature.likes:
                file_handler.write('{} {}\n'.format(
                    index_dict[feature.user_id],
                    index_dict[like_id]
                ))
