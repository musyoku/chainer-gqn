import random
from .. import color, geometry
from .objects import create_object

try:
    from .. import three
except:
    three = None


def generate_blocks(num_blocks=7,
                    color_hue_range=(0, 1),
                    color_saturation_range=(0.75, 1)):
    assert num_blocks > 0
    current_position = (0, 0, 0)
    block_positions = [current_position]
    for _ in range(num_blocks - 1):
        axis = random.choice([0, 1, 2])
        direction = random.choice([-1, 1])
        offset = [0, 0, 0]
        offset[axis] = direction
        new_position = (offset[0] + current_position[0],
                        offset[1] + current_position[1],
                        offset[2] + current_position[2])
        block_positions.append(new_position)
        current_position = new_position

    ret = []
    center_of_gravity = [0, 0, 0]
    for position in block_positions:
        obj, _ = create_object(
            "cube",
            color=color.random_color(
                alpha=1,
                hue_range=color_hue_range,
                saturation_range=color_saturation_range))
        shift = 1
        location = (shift * position[0], shift * position[1],
                    shift * position[2])
        ret.append((obj, location))
        center_of_gravity[0] += location[0]
        center_of_gravity[1] += location[1]
        center_of_gravity[2] += location[2]
    center_of_gravity[0] /= num_blocks
    center_of_gravity[1] /= num_blocks
    center_of_gravity[2] /= num_blocks
    return ret, center_of_gravity


def build_scene(num_blocks=7,
                object_color_hue_range=(0, 1),
                object_color_saturation_range=(0.75, 1)):
    scene = three.Scene()
    blocks, center_of_gravity = generate_blocks(
        num_blocks,
        color_hue_range=object_color_hue_range,
        color_saturation_range=object_color_saturation_range)
    objects = []
    for block in blocks:
        obj, location = block
        position = (location[0] - center_of_gravity[0],
                    location[1] - center_of_gravity[1],
                    location[2] - center_of_gravity[2])
        scene.add(obj, position=position)
        objects.append(obj)
    return scene, objects
