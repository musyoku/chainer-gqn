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
    screen_size = (args.image_size, args.image_size)  # (width, height)
    camera = gqn.three.PerspectiveCamera(
        eye=(3, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 4.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    if args.with_visualization:
        figure = gqn.imgplot.Figure()
        axis = gqn.imgplot.ImageData(screen_size[0], screen_size[1], 3)
        figure.add(axis, 0, 0, 1, 1)
        window = gqn.imgplot.Window(figure, (800, 800))
        window.show()

    image = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])
    dataset = gqn.data.Archiver(
        path=args.path,
        total_observations=args.total_observations,
        num_observations_per_file=args.num_observations_per_file,
        image_size=(args.image_size, args.image_size))

    tick = 0
    start = time.time()
    while True:
        scene, _, _ = gqn.environment.room.build_scene(
            object_names=["cube", "sphere", "cone", "cylinder", "icosahedron"],
            num_objects=random.choice([x for x in range(1, 6)]))
        renderer.set_scene(scene)

        scene_data = gqn.data.archiver.SceneData(screen_size,
                                                 args.num_views_per_scene)

        for _ in range(args.num_views_per_scene):
            eye = (random.uniform(-3, 3), 1, random.uniform(-3, 3))
            center = (random.uniform(-3, 3), random.uniform(0, 1),
                      random.uniform(-3, 3))
            yaw = gqn.math.yaw(eye, center)
            pitch = gqn.math.pitch(eye, center)
            camera.look_at(
                eye=eye,
                center=center,
                up=(0.0, 1.0, 0.0),
            )
            renderer.render(camera, image)

            # [0, 255] -> [-1, 1]
            normalized_image = (image / 255 - 0.5) * 2.0

            scene_data.add(normalized_image, eye, math.cos(yaw), math.cos(yaw),
                           math.sin(pitch), math.sin(pitch))

            if args.with_visualization:
                axis.update(np.uint8(image))

            if args.with_visualization and window.closed():
                return

        tick += 1
        if tick % 5000 == 0:
            print("{} / {} fps:{}".format(tick, args.total_observations,
                                          int(tick / (time.time() - start))))

        dataset.add(scene_data)

        if tick >= args.total_observations:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-visualization",
        "-visualize",
        action="store_true",
        default=False)
    parser.add_argument(
        "--with-object-rotations",
        "-rotate-object",
        action="store_true",
        default=False)
    parser.add_argument(
        "--total-observations", "-total", type=int, default=2000000)
    parser.add_argument(
        "--num-observations-per-file", "-per-file", type=int, default=2000)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--path", type=str, default="rooms_dataset")
    args = parser.parse_args()
    main()
