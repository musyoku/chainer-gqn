import numpy as np
from .. import rasterizer


def render_depth_map(scene, screen_size=(64, 64)):
    depth_map = np.zeros(screen_size, dtype="float32")
    face_index_map = np.zeros(screen_size, dtype="int32")
    object_index_map = np.zeros(screen_size, dtype="int32")
    for object_index, obj in enumerate(scene.objects):
        rasterizer.update_depth_map(obj.face_vertices, face_index_map,
                                    depth_map)
    return depth_map