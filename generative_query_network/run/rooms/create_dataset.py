import argparse
import math
import time
import sys
import os
import random
import numpy as np

sys.path.append(os.path.join("..", ".."))
import gqn


def main():
    screen_size = (64, 64)  # (width, height)
    camera = gqn.three.PerspectiveCamera(
        eye=(3, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 2.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    figure = gqn.viewer.Figure()
    axis = gqn.viewer.ImageData(screen_size[0], screen_size[1], 3)
    figure.add(axis, 0, 0, 1, 1)
    window = gqn.viewer.Window(figure, (800, 800))
    window.show()

    frame = np.zeros(screen_size + (3, ), dtype="uint32")

    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    tick = 0
    start = time.time()
    while True:
        scene, _, _ = gqn.environment.room.build_scene(
            object_names=["cube", "sphere", "bunny", "teapot"],
            num_objects=random.choice([x for x in range(1, 6)]))

        renderer.set_scene(scene)

        total_frames = 5
        for _ in range(total_frames):
            rad = random.uniform(0, math.pi * 2)

            camera.look_at(
                eye=(3.0 * math.cos(rad), 1, 3.0 * math.sin(rad)),
                center=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
            )
            renderer.render(camera, frame)

            axis.update(np.uint8(frame))

            tick += 1
            if tick % 1000 == 0:
                print(tick / (time.time() - start))
            if window.closed():
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
