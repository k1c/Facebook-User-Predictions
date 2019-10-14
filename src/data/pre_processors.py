import os.path
import pandas as pd
import numpy as np


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
