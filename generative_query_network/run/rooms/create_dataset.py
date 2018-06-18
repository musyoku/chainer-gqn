import argparse
import math
import time
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
import gqn


def main():
    screen_size = (128, 128)  # (width, height)
    scene, room, objects = gqn.environment.room.build_scene(
        object_names=["cube", "sphere", "bunny", "teapot"])
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
            eye=(3.0 * math.cos(rad), 1, 3.0 * math.sin(rad)),
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
