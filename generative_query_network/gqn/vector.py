import numpy as np
from chainer import functions

def normalize(vector, xp=None):
    xp = np if xp is None else xp
    return vector / xp.linalg.norm(vector)