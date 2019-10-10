import datetime
import uuid
import json


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def get_random_id():
    return uuid.uuid4().hex


def read_config_file(config_path):
    with open(config_path) as json_data_file:
        data = json.load(json_data_file)
    return data
