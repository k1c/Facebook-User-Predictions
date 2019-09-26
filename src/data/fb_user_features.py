from typing import List


class FBUserFeatures:
    def __init__(self, user_id: str, likes: List[int], statuses: List[str]):
        self.user_id = user_id
        self.likes = likes
        self.statuses = statuses
        self.image = self._load_image(user_id)

    # TODO: Change the return type to the actual thing. The `int` is just a placeholder.
    def _load_image(self, user_id: str) -> int:
        pass
