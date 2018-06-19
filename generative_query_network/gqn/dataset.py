import numpy as np
import os


class Dataset:
    def __init__(self,
                 path,
                 total_observations=2000000,
                 num_observations_per_file=2000,
                 image_size=(64, 64)):
        assert path is not None
        self.images = np.zeros(
            (num_observations_per_file, ) + image_size + (3, ),
            dtype="float32")
        self.viewpoints = np.zeros(
            (num_observations_per_file, 5), dtype="float32")
        self.current_num_observations = 0
        self.current_pool_index = 0
        self.current_file_number = 1
        self.total_observations = total_observations
        self.num_observations_per_file = num_observations_per_file
        self.path = path
        self.image_size = image_size

        try:
            os.mkdir(path)
        except:
            pass

    def add(self, image, camera_position, camera_yaw_rad, camera_pitch_rad):
        assert isinstance(image, np.ndarray)
        assert isinstance(camera_position, tuple)
        assert isinstance(camera_yaw_rad, float)
        assert isinstance(camera_pitch_rad, float)
        assert image.ndim == 3
        assert image.shape[0] == self.image_size[0]
        assert image.shape[1] == self.image_size[1]
        assert image.shape[2] == 3
        assert len(camera_position) == 3

        self.images[self.current_pool_index] = image
        self.viewpoints[self.current_pool_index] = (camera_position[0],
                                                    camera_position[1],
                                                    camera_position[2],
                                                    camera_yaw_rad,
                                                    camera_pitch_rad)
        self.current_pool_index += 1
        if self.current_pool_index >= self.num_observations_per_file:
            self.save()
            self.current_pool_index = 0
            self.current_file_number += 1

    def save(self):
        filename = "images-{:04d}-of-{}.npy".format(
            self.current_file_number, self.num_observations_per_file)
        np.save(os.path.join(self.path, filename), self.images)
        filename = "viewpoints-{:04d}-of-{}.npy".format(
            self.current_file_number, self.num_observations_per_file)
        np.save(os.path.join(self.path, filename), self.viewpoints)
