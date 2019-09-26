from typing import List


class FBUserFeatures:
    def __init__(self, user_id: str, likes: List[int], statuses: List[str], image: int):
        self.user_id = user_id
        self.likes = likes
        self.statuses = statuses
        # TODO: Load actual image. The total size of the images is ~100 mb. Seems ok to load everything to memory.
        self.image = image

    @classmethod
    def from_data(cls, user_id: str, likes: List[int], statuses: List[str], image: int) -> 'FBUserFeatures':
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
        statuses: {}
        """.format(self.user_id, self.likes, self.statuses)
