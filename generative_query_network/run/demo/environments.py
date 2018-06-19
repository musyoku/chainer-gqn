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
        eye=(3, 1, 0),
        center=(0, 0.5, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 1.5,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.01,
        z_far=100)

    figure = gqn.imgplot.Figure()
    axis_room = gqn.imgplot.ImageData(screen_size[0], screen_size[1], 3)
    axis_shepard_matzler = gqn.imgplot.ImageData(screen_size[0], screen_size[1],
                                                3)
    figure.add(axis_room, 0, 0, 0.5, 1)
    figure.add(axis_shepard_matzler, 0.5, 0, 0.5, 1)
    window = gqn.imgplot.Window(figure, (1600, 800))
    window.show()

    image_room = np.zeros(screen_size + (3, ), dtype="uint32")
    image_shepard_matzler = np.zeros(screen_size + (3, ), dtype="uint32")

    renderer_shepard_matzler = gqn.three.Renderer(screen_size[0],
                                                  screen_size[1])
    renderer_room = gqn.three.Renderer(screen_size[0], screen_size[1])

    while True:
        scene_room, _, _ = gqn.environment.room.build_scene(
            object_names=gqn.environment.objects.available_names(),
            num_objects=random.choice([x for x in range(1, 6)]),
            scale_range=(1, 1))
        scene_shepard_metzler, _ = gqn.environment.shepard_metzler.build_scene(
            num_blocks=random.choice([x for x in range(3, 20)]))

        renderer_room.set_scene(scene_room)
        renderer_shepard_matzler.set_scene(scene_shepard_metzler)

        total_frames = 500
        tick = 0
        start = time.time()
        for _ in range(total_frames):
            rad = math.pi * 2 * tick / total_frames

            camera.look_at(
                eye=(3.0 * math.cos(rad), 1, 3.0 * math.sin(rad)),
                center=(0.0, 0.5, 0.0),
                up=(0.0, 1.0, 0.0),
            )
            renderer_room.render(camera, image_room)
            renderer_shepard_matzler.render(camera, image_shepard_matzler)

            axis_room.update(np.uint8(image_room))
            axis_shepard_matzler.update(np.uint8(image_shepard_matzler))

            tick += 1
            if tick % 1000 == 0:
                print(tick / (time.time() - start))

            if window.closed():
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
