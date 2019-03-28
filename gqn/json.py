import os
import json


class JsonSerializable(object):
    def __init__(self):
        self.snapshot_filename = None

    def load(self, root_directory):
        json_path = os.path.join(root_directory, self.snapshot_filename)
        if os.path.exists(json_path) and os.path.isfile(json_path):
            with open(json_path, "r") as f:
                print("Loading", json_path)
                obj = json.load(f)
                for (key, value) in obj.items():
                    if isinstance(value, list):
                        value = tuple(value)
                    setattr(self, key, value)
                return True
        return False

    def save(self, root_directory):
        with open(os.path.join(root_directory, self.snapshot_filename),
                  "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)
