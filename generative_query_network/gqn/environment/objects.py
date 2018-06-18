import os
from .. import geometry, three, color


def load_object(name, color=(1, 1, 1, 1), scale=(1, 1, 1)):
    vertices, faces = geometry.load("../../geometries/{}.obj".format(name))
    return three.Object(faces, vertices, color, scale), vertices, faces


def create_object(name, color=(1, 1, 1, 1), scale=(1, 1, 1)):
    assert name in object_cache
    cached_object, vertices = object_cache[name]
    obj = cached_object.clone()
    obj.set_color(color)
    obj.set_scale(scale)
    return obj, vertices


def generate_wall_colors(num_walls_to_paint=2):
    white = (1, 1, 1, 1)
    if num_walls_to_paint == 0:
        return (white, white, white, white)
    if num_walls_to_paint == 1:
        north_wall_color = color.random_color()
        return (north_wall_color, white, white, white)
    if num_walls_to_paint == 2:
        north_wall_color = color.random_color()
        south_wall_color = color.random_color()
        return (north_wall_color, white, south_wall_color, white)
    if num_walls_to_paint == 3:
        north_wall_color = color.random_color()
        east_wall_color = color.random_color()
        south_wall_color = color.random_color()
        return (north_wall_color, east_wall_color, south_wall_color, white)
    if num_walls_to_paint == 4:
        north_wall_color = color.random_color()
        east_wall_color = color.random_color()
        south_wall_color = color.random_color()
        west_wall_color = color.random_color()
        return (north_wall_color, east_wall_color, south_wall_color,
                west_wall_color)
    assert False


def create_cornell_box(size=(7, 3, 7), num_walls_to_paint=2):
    north_wall_color, east_wall_color, south_wall_color, west_wall_color = generate_wall_colors(
        num_walls_to_paint)
    return three.CornellBox(north_wall_color, east_wall_color,
                            south_wall_color, west_wall_color, size)


# オブジェクト初期化時に法線ベクトルの計算が走る
# 一度生成したオブジェクトをコピーして高速化する
# Object allocation is heavy
# See three/core/scene/object.cpp
object_cache = {}
geometories = os.listdir("../../geometries")
for geometory_filename in geometories:
    object_name = geometory_filename.replace(".obj", "")
    obj, vertices, _ = load_object(object_name)
    object_cache[object_name] = (obj, vertices)
