import random
import colorsys


def random_color(alpha=1):
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.75, 1)
    brightness = 1
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (red, green, blue, alpha)