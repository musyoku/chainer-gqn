import argparse, math
import gqn
import numpy as np


def create_object(name, color=(1, 1, 1, 1), scale=(1, 1, 1)):
    vertices, faces = gqn.geometry.load("geometries/{}.obj".format(name))
    return gqn.three.Object(faces, vertices, color, scale)


def create_cornell_box(size=(7, 3, 7), color=(1, 1, 1, 1)):
    vertices, faces = gqn.geometry.load(
        "geometries/{}.obj".format("cornell_box"))
    return gqn.three.Object(faces, vertices, color, size)


def create_scene():
    scene = gqn.three.Scene()
    objects = []
    room = create_cornell_box(size=(4, 4, 4))
    scene.add(room, position=(0, 1, 0))

    wall = create_object("cube", scale=(7.0, 3.0, 1.0))
    # scene.add(wall, position=(0.0, 0.0, 3.0))

    # wall = create_object("cube", scale=(7.0, 3.0, 1.0))
    # scene.add(wall, position=(0.0, 0.0, -3.0))

    # wall = create_object("cube", scale=(7.0, 3.0, 1.0))
    # scene.add(wall, position=(3.0, 0.0, 0.0), rotation=(0, math.pi / 2.0, 0.0))

    # wall = create_object("cube", scale=(7.0, 3.0, 1.0))
    # scene.add(wall, position=(0.0, 0.0, 3.0), rotation=(0, math.pi / 2.0, 0.0))

    obj = create_object("cube", scale=(0.5, 0.5, 0.5))
    # scene.add(obj, position=(0.5, 0.0, 1.0))
    objects.append(obj)
    obj = create_object("polyhedron", scale=(0.5, 0.5, 0.5))
    # scene.add(obj, position=(-1.0, 0.0, -0.5))
    objects.append(obj)
    return scene, room, objects


def main():
    screen_size = (256, 256)  # (width, height)
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])
    scene, room, objects = create_scene()
    camera = gqn.three.PerspectiveCamera(
        eye=(1.0 * math.cos(math.pi * 0.6954166666671697), 0.5,
             1.0 * math.sin(math.pi * 0.6954166666671697)),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 2.5,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    figure = gqn.viewer.Figure()
    axis_depth_map = gqn.viewer.ImageData(screen_size[0], screen_size[1], 1)
    figure.add(axis_depth_map, 0, 0, 1, 1)
    window = gqn.viewer.Window(figure)
    window.show()

    rad = 0
    while True:
        rad += math.pi / 60000

        depth_map = np.full(screen_size, 100.0, dtype="float32")
        face_index_map = np.zeros(screen_size, dtype="int32")
        object_index_map = np.zeros(screen_size, dtype="int32")
        camera.look_at(
            eye=(1.0 * math.cos(rad), 0.5, 1.0 * math.sin(rad)),
            center=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        # print(rad / math.pi)
        renderer.render_depth_map(scene, camera, depth_map)

        # axis_depth_map.update(
        #     np.uint8(np.clip(depth_map, 0.0, 1.0) * 255))
        axis_depth_map.update(
            np.uint8(
                np.clip((1.0 - depth_map) / (1.0 - np.amin(depth_map)), 0.0,
                        1.0) * 255))
        # return
        if window.closed():
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()