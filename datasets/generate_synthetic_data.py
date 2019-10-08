import pathlib
import random
import shutil
import uuid
from typing import List

import pandas as pd


def generate_relations_dir(user_ids: List[str], num_relations: int):
    random_like_ids = [46688639082 + i for i in range(10)]

    user_relations = {
        user_id: [random.choice(random_like_ids) for _ in range(num_relations)]
        for user_id in user_ids
    }

    user_id_entries, like_id_entries = list(), list()

    for user_id in user_relations.keys():
        like_ids = user_relations.get(user_id)
        for like_id in like_ids:
            user_id_entries.append(user_id)
            like_id_entries.append(like_id)

    df = pd.DataFrame({
        "userid": user_id_entries,
        "like_id": like_id_entries
    })

    df.to_csv(
        "synthetic/Relation/Relation.csv"
    )


def generate_profiles_dir(user_ids: List[str]):
    num_users = len(user_ids)
    df = pd.DataFrame({
        "userid": user_ids,
        "age": [float(random.randrange(10, 50)) for _ in range(num_users)],
        "gender": [random.choice([0, 1]) for _ in range(num_users)],
        "ope": [random.uniform(0, 5) for _ in range(num_users)],
        "con": [random.uniform(0, 5) for _ in range(num_users)],
        "ext": [random.uniform(0, 5) for _ in range(num_users)],
        "agr": [random.uniform(0, 5) for _ in range(num_users)],
        "neu": [random.uniform(0, 5) for _ in range(num_users)]
    })

    df.to_csv(
        "synthetic/Profile/Profile.csv"
    )


def generate_images_dir(user_ids: List[str]):
    Profile = pd.read_csv("synthetic/Profile/Profile.csv")

    for user_id in user_ids:
        file_path = "synthetic/Image/{}.jpg".format(user_id)
        print(user_id)
        print(Profile[Profile["userid"] == user_id]["gender"].values)

        if Profile[Profile["userid"] == user_id]["gender"].values == 0:
            shutil.copy("datasets/mark.jpg", file_path)
        else:
            shutil.copy("datasets/priscilla.jpg", file_path)


def generate_status_dir(user_ids: List[str]):
    for user_id in user_ids:
        file_path = "synthetic/Text/{}.txt".format(user_id)
        with open(file_path, "w") as f:
            f.writelines([
                "This is a status\n",
                "Yay another one",
            ])


def generate_user_ids(num_users: int):
    return [uuid.uuid4().hex for _ in range(num_users)]


if __name__ == "__main__":

    for dir_name in ['synthetic/Profile/', 'synthetic/Relation/', 'synthetic/Text/', 'synthetic/Image/']:
        pathlib.Path(dir_name).mkdir(
            parents=True,
            exist_ok=True
        )

    user_ids = generate_user_ids(num_users=50)
    generate_profiles_dir(user_ids)
    generate_relations_dir(user_ids, num_relations=3)
    generate_status_dir(user_ids)
    generate_images_dir(user_ids)
