from typing import List


class UserFeatures:
    def __init__(self, user_id: str, likes: List[int]):
        self.user_id = user_id
        self.likes = likes

    def __repr__(self):
        return """
        user_id: {} \n
        likes: {} \n
        """.format(self.user_id, self.likes)
