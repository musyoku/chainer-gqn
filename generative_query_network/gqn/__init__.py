from . import geometry
from . import vertices
from . import matrix
from . import scene
from . import environment
from . import data
from . import color
from . import nn
from . import mathematics as math

try:
    from . import imgplot
except:
    imgplot = None
    # print("\033[93m")
    # print("imgplot.so not found.")
    # print("Please build imgplot before running your code.")
    # print("`cd imgplot`")
    # print("`make`")
    # print("\033[0m")

try:
    from . import three
except:
    three = None
    # print("\033[93m")
    # print("three.so not found.")
    # print("Please build three before running your code.")
    # print("`cd three`")
    # print("`make`")
    # print("\033[0m")