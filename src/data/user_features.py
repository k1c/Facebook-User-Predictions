from typing import List
import numpy as np


class UserFeatures:
    def __init__(self, user_id: str, likes: List[int], statuses: List[str], image: np.ndarray):
        self.user_id = user_id
        self.likes = likes
        self.statuses = statuses
        self.image = image

    @classmethod
    def from_data(cls, user_id: str, likes: List[int], statuses: List[str], image: np.ndarray) -> 'UserFeatures':
        return cls(
            user_id=user_id,
            likes=likes,
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
