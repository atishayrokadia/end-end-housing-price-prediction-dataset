import os
import sys
from src.mlProject.exception import CustomException

import pickle



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def find_key_by_value(d, target_value):
    for key, value in d.items():
        if value == target_value:
            return key
    return None

