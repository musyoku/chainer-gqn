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
    camera_room = gqn.three.PerspectiveCamera(
        eye=(3, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 4.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)
    camera_shepard_matzler = gqn.three.PerspectiveCamera(
        eye=(6, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 4.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    figure = gqn.imgplot.Figure()
    axes_room = []
    axes_shepard_matzler = []
    for x in range(5):
        for y in range(5):
            axis = gqn.imgplot.ImageData(screen_size[0], screen_size[1], 3)
            axes_room.append(axis)
            figure.add(axis, 0.1 * x, 0.2 * y, 0.1, 0.2)
    for x in range(5):
        for y in range(5):
            axis = gqn.imgplot.ImageData(screen_size[0], screen_size[1], 3)
            axes_shepard_matzler.append(axis)
            figure.add(axis, 0.1 * x + 0.5, 0.2 * y, 0.1, 0.2)
    window = gqn.imgplot.Window(figure, (1600, 800))
    window.show()

    image_room = np.zeros(screen_size + (3, ), dtype="uint32")
    image_shepard_matzler = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer_shepard_matzler = gqn.three.Renderer(screen_size[0],
                                                  screen_size[1])
    renderer_room = gqn.three.Renderer(screen_size[0], screen_size[1])

    while True:
        for scene_idx in range(5):
            scene_room, _, _ = gqn.environment.room.build_scene(
                object_names=[
                    "cube", "sphere", "cone", "cylinder", "icosahedron"
                ],
                num_objects=random.choice([x for x in range(1, 6)]))
            renderer_room.set_scene(scene_room)

            scene_shepard_matzler, _ = gqn.environment.shepard_metzler.build_scene(
                num_blocks=random.choice([x for x in range(3, 14)]))
            renderer_shepard_matzler.set_scene(scene_shepard_matzler)

            for observation_idx in range(5):
                if window.closed():
                    return
                camera_room.look_at(
                    eye=(random.uniform(-3, 3), 1, random.uniform(-3, 3)),
                    center=(random.uniform(-3, 3), random.uniform(0, 1),
                            random.uniform(-3, 3)),
                    up=(0.0, 1.0, 0.0),
                )
                renderer_room.render(camera_room, image_room)
                axes_room[observation_idx * 5 + scene_idx].update(
                    np.uint8(image_room))

            for observation_idx in range(5):
                if window.closed():
                    return
                rad = random.uniform(0, math.pi * 2)
                camera_shepard_matzler.look_at(
                    eye=(6.0 * math.cos(rad), 6.0 * math.sin(rad), 6.0 * math.sin(rad)),
                    center=(0.0, 0.0, 0.0),
                    up=(0.0, 1.0, 0.0),
                )
                renderer_shepard_matzler.render(camera_shepard_matzler,
                                                image_shepard_matzler)
                axes_shepard_matzler[observation_idx * 5 + scene_idx].update(
                    np.uint8(image_shepard_matzler))

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
