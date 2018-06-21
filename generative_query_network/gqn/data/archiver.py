import numpy as np
import os


class SceneData:
    def __init__(self, image_size, num_views):
        self.images = np.zeros(
            (num_views, ) + image_size + (3, ), dtype="float32")
        self.viewpoints = np.zeros((num_views, 7), dtype="float32")
        self.view_index = 0
        self.num_views = num_views
        self.image_size = image_size

    def add(self, image, camera_position, cos_camera_yaw_rad,
            sin_camera_yaw_rad, cos_camera_pitch_rad, sin_camera_pitch_rad):
        assert isinstance(image, np.ndarray)
        assert isinstance(camera_position, tuple)
        assert isinstance(cos_camera_yaw_rad, float)
        assert isinstance(sin_camera_yaw_rad, float)
        assert isinstance(cos_camera_pitch_rad, float)
        assert isinstance(sin_camera_pitch_rad, float)
        assert image.ndim == 3
        assert image.shape[0] == self.image_size[0]
        assert image.shape[1] == self.image_size[1]
        assert image.shape[2] == 3
        assert len(camera_position) == 3
        assert self.view_index < self.num_views

        self.images[self.view_index] = image
        self.viewpoints[self.view_index] = (
            camera_position[0],
            camera_position[1],
            camera_position[2],
            cos_camera_yaw_rad,
            sin_camera_yaw_rad,
            cos_camera_pitch_rad,
            sin_camera_pitch_rad,
        )
        self.view_index += 1


class Archiver:
    def __init__(self,
                 path,
                 total_observations=2000000,
                 num_observations_per_file=2000,
                 image_size=(64, 64),
                 num_views=5):
        assert path is not None
        self.images = np.zeros(
            (num_observations_per_file, num_views) + image_size + (3, ),
            dtype="float32")
        self.viewpoints = np.zeros(
            (num_observations_per_file, num_views, 7), dtype="float32")
        self.current_num_observations = 0
        self.current_pool_index = 0
        self.current_file_number = 1
        self.total_observations = total_observations
        self.num_observations_per_file = num_observations_per_file
        self.path = path
        self.image_size = image_size
        self.num_views = num_views

        try:
            os.mkdir(path)
        except:
            pass
        try:
            os.mkdir(os.path.join(path, "images"))
        except:
            pass
        try:
            os.mkdir(os.path.join(path, "viewpoints"))
        except:
            pass

    def add(self, scene: SceneData):
        assert isinstance(scene, SceneData)

        self.images[self.current_pool_index] = scene.images
        self.viewpoints[self.current_pool_index] = scene.viewpoints
        
        self.current_pool_index += 1
        if self.current_pool_index >= self.num_observations_per_file:
            self.save()
            self.current_pool_index = 0
            self.current_file_number += 1

    def save(self):
        filename = "{:03d}-of-{}.npy".format(self.current_file_number,
                                             self.num_observations_per_file)
        np.save(os.path.join(self.path, "images", filename), self.images)
        filename = "{:03d}-of-{}.npy".format(self.current_file_number,
                                             self.num_observations_per_file)
        np.save(
            os.path.join(self.path, "viewpoints", filename), self.viewpoints)
