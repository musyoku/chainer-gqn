import random
import colorsys


def random_color(alpha=1, hue_range=(0, 1), saturation_range=(0.75, 1)):
    hue = random.uniform(hue_range[0], hue_range[1])
    saturation = random.uniform(saturation_range[0], saturation_range[1])
    brightness = 1
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (red, green, blue, alpha)