import argparse, math
import gqn
import numpy as np


def create_object(name, color=(1, 1, 1, 1)):
    vertices, faces = gqn.geometry.load("geometries/{}.obj".format(name))
    return gqn.three.Object(faces, vertices, color)


def create_scene():
    scene = gqn.three.Scene()
    objects = []
    obj = create_object("teapot")
    scene.add(obj)
    objects.append(obj)
    return scene, objects


def main():
    screen_size = (64, 64)
    scene, objects = create_scene()
    camera = gqn.three.PerspectiveCamera(
        eye=(2.0, 2.0, 2.0),
        center=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_rad=math.pi * 2.0 / 3.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=100)

    figure = gqn.viewer.Figure()
    axis_depth_map = gqn.viewer.ImageData(screen_size[0], screen_size[1], 1)
    figure.add(axis_depth_map, 0, 0, 1, 1)
    window = gqn.viewer.Window(figure)
    window.show()

    rad = 0
    while True:
        rad += math.pi / 2500
        depth_map = np.full(screen_size, 1.0, dtype="float32")
        face_index_map = np.zeros(screen_size, dtype="int32")
        object_index_map = np.zeros(screen_size, dtype="int32")
        camera.look_at(
            eye=(1.0 * math.sin(rad), 1.0, 1.0 * math.cos(rad)),
            center=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        gqn.renderer.render_depth_map(scene, camera, face_index_map,
                                      object_index_map, depth_map)
        axis_depth_map.update(
            np.uint8((1.0 - depth_map) / (1.0 - np.amin(depth_map)) * 255))
        if window.closed():
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()