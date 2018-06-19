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
    screen_size = (640, 640)  # (width, height)
    camera = gqn.three.PerspectiveCamera(
        eye=(2, 0.25, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 4,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.01,
        z_far=100)

    figure = gqn.viewer.Figure()
    axis = gqn.viewer.ImageData(screen_size[0], screen_size[1], 3)
    figure.add(axis, 0, 0, 1, 1)
    window = gqn.viewer.Window(figure, (640, 640))
    window.show()

    image = np.zeros(screen_size + (3, ), dtype="uint32")

    scene = gqn.three.Scene()
    room = gqn.environment.objects.create_cornell_box(
        size=(2, 2, 2), num_walls_to_paint=2)
    scene.add(room, position=(0, 0, 0))

    object_names = gqn.environment.objects.available_names()
    for idx, name in enumerate(object_names):
        obj, _, _ = gqn.environment.objects.load_object(
            name, color=gqn.color.random_color(), scale=(0.25, 0.25, 0.25))
        offset = (0.5, 0, -0.5)
        scene.add(obj, position=(offset[idx % 3], offset[idx // 3], 0))

    renderer = gqn.three.Renderer(scene, screen_size[0], screen_size[1])

    while True:
        total_frames = 1000
        tick = 0
        for _ in range(total_frames):
            rad = math.pi * 2 * tick / total_frames
            camera.look_at(
                eye=(2.0 * math.cos(rad), 0.25, 2.0 * math.sin(rad)),
                center=(0.0, 0, 0.0),
                up=(0.0, 1.0, 0.0),
            )
            renderer.render(camera, image)
            axis.update(np.uint8(image))
            tick += 1

            if window.closed():
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
