import os.path
import pandas as pd
from data.user_features import UserFeatures
from typing import List
import subprocess
import os
import numpy as np
from util.utils import get_random_id
import itertools
from constants.directory_names import TEMP_DIR


def pre_process_likes_v1(data_path: str) -> pd.DataFrame:
    def load_likes_csv_file(file_path):
        return pd.read_csv(
            file_path
        ).drop(
            columns=['Unnamed: 0']
        ).sort_values(
            by=['userid']
        )

    def get_user_ids(data_frame):
        return data_frame['user_id']

    def get_page_ids_liked_by_user(data_frame, user_id):
        return data_frame[data_frame['user_id'] == user_id]['like_id']

    def get_page_total_likes(page_id):
        return like_counts_per_page[page_id]

    original_csv_file_path = os.path.join(data_path, 'Relation', 'Relation.csv')
    preprocessed_csv_file_path = os.path.join(data_path, 'Relation', 'relation_preprocessed_raw_v1.csv')

    if os.path.isfile(preprocessed_csv_file_path):
        features = load_likes_csv_file(preprocessed_csv_file_path)
    else:
        relation_df = load_likes_csv_file(original_csv_file_path)
        like_counts_per_user = relation_df['user_id'].value_counts()
        like_counts_per_page = relation_df['like_id'].value_counts()
        features = relation_df.assign(
            user_id=like_counts_per_user.keys(),
            likes_given=like_counts_per_user.values
        )
        features = features.assign(
            pages_liked_sum_likes=np.array([
                np.array([
                    get_page_total_likes(page_id) for page_id in get_page_ids_liked_by_user(relation_df, user_id)
                ]).sum()
                for user_id in get_user_ids(features)
            ])
        )

    # Standardize features by removing the mean and scaling to unit variance
    features[features.columns[1:]] = features[features.columns[1:]].apply(
        lambda df: (df-df.mean())/df.std()
    ).fillna(0)

    return features


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
