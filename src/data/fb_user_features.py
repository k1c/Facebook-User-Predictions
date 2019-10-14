from typing import List
import numpy as np


class FBUserFeatures:
    def __init__(
            self,
            user_id: str,
            likes: List[int],
            likes_preprocessed_v1: List[float],
            statuses: List[str],
            image: np.ndarray
    ):
        self.user_id = user_id
        self.likes = likes
        self.likes_preprocessed_v1 = likes_preprocessed_v1
        self.statuses = statuses
        self.image = image

    @classmethod
    def from_data(
            cls,
            user_id: str,
            likes: List[int],
            likes_preprocessed_v1: List[float],
            statuses: List[str],
            image: np.ndarray
    ) -> 'FBUserFeatures':
        return cls(
            user_id=user_id,
            likes=likes,
            likes_preprocessed_v1=likes_preprocessed_v1,
            statuses=statuses,
            image=image
        )

    def __repr__(self):
        return """
        user_id: {} \n
        likes: {} \n
        statuses: {} \n
        image: {}
        """.format(self.user_id, self.likes, self.statuses, self.image)
