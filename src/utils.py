import datetime
import uuid


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def get_random_id():
    return uuid.uuid4().hex
