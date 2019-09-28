import datetime
import os
import errno


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def create_directories_to_file(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
