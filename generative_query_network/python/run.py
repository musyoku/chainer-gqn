import argparse, math
import gqn
import numpy as np


def create_object(name):
    vertices, faces = gqn.geometry.load("geometries/{}.obj".format(name))
    face_vertices = gqn.vertices.convert_to_face_representation(
        vertices, faces)
    obj = gqn.scene.Object(faces, vertices, face_vertices)
    return obj


def create_scene():
    scene = gqn.scene.Scene()
    scene.add(create_object("teapot"))
    return scene


def main():
    scene = create_scene()

    # グローバル座標系からカメラ座標系に変換する行列
    mat_view = gqn.matrix.look_at(
        np.asarray([0.0, 0.0, 3.0]), np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]), np)
    # 鏡像変換行列
    mat_mirror = np.eye(4, dtype="float32")
    mat_mirror[2, 2] = -1
    # 射影行列
    mat_perspective = gqn.matrix.perspective(math.pi * 2.0 / 3.0, 1.0, 0.1,
                                             100.0)

    for object_index, obj in enumerate(scene.objects):
        vertices = np.copy(obj.vertices)
        vertices = vertices @ mat_view
        vertices = vertices @ mat_mirror
        vertices = vertices @ mat_perspective
        w = vertices[..., 3, None]
        vertices = vertices / w

        face_vertices = gqn.vertices.convert_to_face_representation(
            vertices, obj.faces)
        obj.face_vertices = face_vertices

    figure = gqn.viewer.Figure()

    screen_size = (64, 64)
    axis_depth_map = gqn.viewer.ImageData(screen_size[0], screen_size[1], 1)
    figure.add(axis_depth_map, 0, 0, 1, 1)

    window = gqn.viewer.Window(figure)
    window.show()

    depth_map = gqn.renderer.render_depth_map(scene, screen_size)
    axis_depth_map.update(np.uint8((1.0 - depth_map) / (1.0 - np.amin(depth_map)) * 255))

    while True:
        if window.closed():
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()