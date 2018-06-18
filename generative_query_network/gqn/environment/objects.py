import os
from .. import geometry, three


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


def create_cornell_box(size=(7, 3, 7), color=(1, 1, 1, 1)):
    vertices, faces = geometry.load(
        "../../geometries/{}.obj".format("cornell_box"))
    return three.Object(faces, vertices, color, size)


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
