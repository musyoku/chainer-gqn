import argparse, math, time
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
    room = create_cornell_box(size=(7, 4, 7))
    scene.add(room, position=(0, 1.5, 0))

    obj = create_object(
        "teapot", scale=(0.5, 0.5, 0.5), color=(1.0, 0.0, 1.0, 1.0))
    scene.add(obj, position=(1.0, 0.0, 1.0))
    objects.append(obj)
    obj = create_object(
        "bunny", scale=(0.5, 0.5, 0.5), color=(1.0, 1.0, 0.0, 1.0))
    scene.add(obj, position=(1.0, 0.0, -1.0))
    objects.append(obj)
    obj = create_object(
        "polyhedron", scale=(0.5, 0.5, 0.5), color=(0.0, 1.0, 1.0, 1.0))
    scene.add(obj, position=(-1.0, 0.0, -1.0))
    objects.append(obj)
    return scene, room, objects


def main():
    screen_size = (320, 320)  # (width, height)
    scene, room, objects = create_scene()
    camera = gqn.three.PerspectiveCamera(
        eye=(1.0 * math.cos(math.pi * 0.6954166666671697), 0.5,
             1.0 * math.sin(math.pi * 0.6954166666671697)),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 2.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    figure = gqn.viewer.Figure()
    axis_depth_map = gqn.viewer.ImageData(screen_size[0], screen_size[1], 1)
    axis_rgb_map = gqn.viewer.ImageData(screen_size[0], screen_size[1], 3)
    figure.add(axis_depth_map, 0, 0, 0.5, 1)
    figure.add(axis_rgb_map, 0.5, 0, 0.5, 1)
    window = gqn.viewer.Window(figure, (1600, 800))
    window.show()

    renderer = gqn.three.Renderer(scene, screen_size[0], screen_size[1])

    rad = 0
    start = time.time()
    num_generated = 0
    depth_map = np.zeros(screen_size, dtype="float32")
    rgb_map = np.zeros(screen_size + (3, ), dtype="uint32")
    while True:
        rad += math.pi * 2 / 10000

        camera.look_at(
            eye=(2.0 * math.cos(rad), 0.5, 2.0 * math.sin(rad)),
            center=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        # print(rad / math.pi)
        renderer.render_depth_map(camera, depth_map)
        renderer.render(camera, rgb_map)

        # axis_depth_map.update(
        #     np.uint8(np.clip(depth_map, 0.0, 1.0) * 255))
        axis_depth_map.update(
            np.uint8(
                np.clip(
                    (1.0 - depth_map) / 0.055428266525268555, 0.0, 1.0) * 255))
        axis_rgb_map.update(np.uint8(rgb_map))
        # return
        if window.closed():
            return

        num_generated += 1
        if num_generated % 1000 == 0:
            print(num_generated / (time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()