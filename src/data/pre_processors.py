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
import networkx as nx
from node2vec import Node2Vec


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


def get_deep_walk_embeddings(features: List[UserFeatures], hyper_params: str):
    input_edge_list_file = '{}/deepwalk_relations_edge_list_{}.txt'.format(
        TEMP_DIR, get_random_id()
    )
    output_embeddings_file = '{}/deepwalk_relations_embeddings_{}.txt'.format(
        TEMP_DIR, get_random_id()
    )
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    create_edge_list_file(input_edge_list_file, features)

    # This library assumes it is used as a process, not a library. Create embeddings:
    print("Running the DeepWalk algorithm to produce embeddings for user likes...")
    subprocess.Popen(
        ' '.join([
            'deepwalk',
            '--input {}'.format(input_edge_list_file),
            '--output {}'.format(output_embeddings_file),
            '--format edgelist',
            '--workers 16',
            hyper_params
        ]),
        shell=True
    ).wait()

    embeddings = np.genfromtxt(output_embeddings_file, delimiter=' ', skip_header=1)
    embeddings_sorted = embeddings[embeddings[:, 0].argsort()]
    os.remove(input_edge_list_file)
    os.remove(output_embeddings_file)

    return embeddings_sorted[:len(features), 1:]


def get_node2vec_embeddings(features: List[UserFeatures], hyper_params: dict):
    input_edge_list_file = '{}/node2vec_relations_edge_list_{}.txt'.format(
        TEMP_DIR, get_random_id()
    )
    output_embeddings_file = '{}/node2vec_relations_node2vec_embeddings_{}.txt'.format(
        TEMP_DIR, get_random_id()
    )
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    create_edge_list_file(input_edge_list_file, features)
    graph = nx.Graph()
    with open(input_edge_list_file, 'r') as file_handler:
        for line in file_handler:
            if line:
                node1, node2 = line.split()
                graph.add_edge(node1, node2)

    node2vec = Node2Vec(graph, workers=4, temp_folder=TEMP_DIR, **hyper_params)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    model.wv.save_word2vec_format(output_embeddings_file)

    embeddings = np.genfromtxt(output_embeddings_file, delimiter=' ', skip_header=1)
    embeddings_sorted = embeddings[embeddings[:, 0].argsort()]
    os.remove(input_edge_list_file)
    os.remove(output_embeddings_file)

    return embeddings_sorted[:len(features), 1:]


def create_edge_list_file(file_name: str, features: List[UserFeatures]):
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
