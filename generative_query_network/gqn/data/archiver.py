import os
import numpy as np
import cupy as cp


class SceneData:
    def __init__(self, image_size, num_views_per_scene):
        self.images = np.zeros(
            (num_views_per_scene, ) + image_size + (3, ), dtype="float32")
        self.viewpoints = np.zeros((num_views_per_scene, 7), dtype="float32")
        self.view_index = 0
        self.num_views_per_scene = num_views_per_scene
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
        assert self.view_index < self.num_views_per_scene

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
                 directory,
                 total_observations=2000000,
                 num_observations_per_file=2000,
                 image_size=(64, 64),
                 num_views_per_scene=5,
                 start_file_number=1):
        assert directory is not None
        self.images = np.zeros(
            (num_observations_per_file, num_views_per_scene) + image_size +
            (3, ),
            dtype="float32")
        self.viewpoints = np.zeros(
            (num_observations_per_file, num_views_per_scene, 7),
            dtype="float32")
        self.current_num_observations = 0
        self.current_pool_index = 0
        self.current_file_number = start_file_number
        self.total_observations = total_observations
        self.num_observations_per_file = num_observations_per_file
        self.directory = directory
        self.image_size = image_size
        self.num_views_per_scene = num_views_per_scene

        self.total_images = 0
        self.dataset_mean = None
        self.dataset_var = None

        try:
            os.mkdir(directory)
        except:
            pass
        try:
            os.mkdir(os.path.join(directory, "images"))
        except:
            pass
        try:
            os.mkdir(os.path.join(directory, "viewpoints"))
        except:
            pass

    def add(self, scene: SceneData):
        assert isinstance(scene, SceneData)

        self.images[self.current_pool_index] = scene.images
        self.viewpoints[self.current_pool_index] = scene.viewpoints

        self.current_pool_index += 1
        if self.current_pool_index >= self.num_observations_per_file:
            self.save_subset()
            self.save_mean_and_variance()
            self.current_pool_index = 0
            self.current_file_number += 1
            self.total_images += self.num_observations_per_file

    def save_mean_and_variance(self):
        subset_size = self.num_observations_per_file
        new_total_size = self.total_images + subset_size
        co1 = self.total_images / new_total_size
        co2 = subset_size / new_total_size

        images = cp.asarray(self.images)

        subset_mean = cp.mean(images, axis=(0, 1))
        subset_var = cp.var(images, axis=(0, 1))

        new_dataset_mean = subset_mean if self.dataset_mean is None else co1 * self.dataset_mean + co2 * subset_mean
        new_dataset_var = subset_var if self.dataset_var is None else co1 * (
            self.dataset_var + self.dataset_mean**2) + co2 * (
                subset_var + subset_mean**2) - new_dataset_mean**2

        # avoid negative value
        new_dataset_var[new_dataset_var < 0] = 0

        self.dataset_var = new_dataset_var
        self.dataset_mean = new_dataset_mean
        self.dataset_std = cp.sqrt(self.dataset_var)

        cp.save(os.path.join(self.directory, "mean.npy"), self.dataset_mean)
        cp.save(os.path.join(self.directory, "std.npy"), self.dataset_std)

    def save_subset(self):
        filename = "{:03d}.npy".format(self.current_file_number)
        np.save(os.path.join(self.directory, "images", filename), self.images)

        filename = "{:03d}.npy".format(self.current_file_number)
        np.save(
            os.path.join(self.directory, "viewpoints", filename),
            self.viewpoints)
