import os
from .. import geometry, three, color


def generate_donut():
    pass


def get_realpath():
    return os.path.realpath(__file__).replace("/gqn/environment/objects.py",
                                              "")


def load_object(name, color=(1, 1, 1, 1), scale=(1, 1, 1)):
    vertices, faces = geometry.load("{}/geometries/{}".format(
        get_realpath(), name))
    num_vertices = vertices.shape[0]
    smoothness = True if num_vertices > 20 else False
    return three.Object(faces, vertices, color, scale,
                        smoothness), vertices, faces


def create_object(name, color=(1, 1, 1, 1), scale=(1, 1, 1)):
    assert name in object_cache
    cached_object, vertices = object_cache[name]
    obj = cached_object.clone()
    obj.set_color(color)
    obj.set_scale(scale)
    return obj, vertices


def generate_wall_colors(num_walls_to_paint=2, saturation_range=(0.5, 0.75)):
    white = (1, 1, 1, 1)
    if num_walls_to_paint == 0:
        return (white, white, white, white)
    if num_walls_to_paint == 1:
        north_wall_color = color.random_color(
            saturation_range=saturation_range)
        return (north_wall_color, white, white, white)
    if num_walls_to_paint == 2:
        north_wall_color = color.random_color(
            saturation_range=saturation_range)
        south_wall_color = color.random_color(
            saturation_range=saturation_range)
        return (north_wall_color, white, south_wall_color, white)
    if num_walls_to_paint == 3:
        north_wall_color = color.random_color(
            saturation_range=saturation_range)
        east_wall_color = color.random_color(saturation_range=saturation_range)
        south_wall_color = color.random_color(
            saturation_range=saturation_range)
        return (north_wall_color, east_wall_color, south_wall_color, white)
    if num_walls_to_paint == 4:
        north_wall_color = color.random_color(
            saturation_range=saturation_range)
        east_wall_color = color.random_color(saturation_range=saturation_range)
        south_wall_color = color.random_color(
            saturation_range=saturation_range)
        west_wall_color = color.random_color(saturation_range=saturation_range)
        return (north_wall_color, east_wall_color, south_wall_color,
                west_wall_color)
    assert False


def create_cornell_box(size=(7, 3, 7), num_walls_to_paint=2):
    north_wall_color, east_wall_color, south_wall_color, west_wall_color = generate_wall_colors(
        num_walls_to_paint)
    return three.CornellBox(north_wall_color, east_wall_color,
                            south_wall_color, west_wall_color, size)


def available_names():
    return list(object_cache.keys())


# オブジェクト初期化時に法線ベクトルの計算が走る
# 一度生成したオブジェクトをコピーして高速化する
# Object allocation is heavy
# See three/core/scene/object.cpp
object_cache = {}
geometories = os.listdir("{}/geometries".format(get_realpath()))
for object_name in geometories:
    obj, vertices, _ = load_object(object_name)
    object_cache[object_name] = (obj, vertices)
