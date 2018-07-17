import random
import math
import numpy as np
from .. import geometry, color
try:
    from .. import three
except:
    pass
from .objects import create_cornell_box, create_object


def build_scene(
        room_size=(7, 4, 7),
        num_objects=3,
        object_names=["cube"],
        scale_range=(0.5, 1.0),
        num_walls_to_paint=2,
        object_placement_radius=3,
        object_color_hue_range=(0, 1),
        object_color_saturation_range=(0.75, 1)):
    scene = three.Scene()
    room = create_cornell_box(
        size=room_size, num_walls_to_paint=num_walls_to_paint)
    room_offset = room_size[1] / 2.0
    scene.add(room, position=(0, room_offset, 0))
    objects = []

    for _ in range(num_objects):
        objct_name = random.choice(object_names)
        scale = random.uniform(scale_range[0], scale_range[1])
        obj, vertices = create_object(
            objct_name,
            scale=(scale, scale, scale),
            color=color.random_color(
                hue_range=object_color_hue_range,
                saturation_range=object_color_saturation_range))
        offset_y = np.amin(vertices, axis=0)[1] * scale
        theta = random.uniform(0, 2) * math.pi
        radius = random.uniform(0.5, 1) * object_placement_radius
        scene.add(
            obj,
            position=(radius * math.cos(theta), -offset_y,
                      radius * math.sin(theta)))
        objects.append(obj)
    return scene, room, objects