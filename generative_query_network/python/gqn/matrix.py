import numpy as np
import math
from .vector import normalize

def rotation_x(rad, xp=None):
    xp = np if xp is None else xp
    return xp.asarray([
        [1, 0, 0, 0],
        [0, math.cos(rad), -math.sin(rad), 0],
        [0, math.sin(rad), math.cos(rad), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(rad, xp=None):
    xp = np if xp is None else xp
    return xp.asarray([
        [math.cos(rad), 0, math.sin(rad), 0],
        [0, 1, 0, 0],
        [-math.sin(rad), 0, math.cos(rad), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(rad, xp=None):
    xp = np if xp is None else xp
    return xp.asarray([
        [math.cos(rad), -math.sin(rad), 0, 0],
        [math.sin(rad), math.cos(rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def perspective(fovy_rad, aspect, z_near, z_far, xp=None):
    xp = np if xp is None else xp
    tan_half_fovy = math.tan(fovy_rad / 2.0)
    mat = xp.zeros((4, 4), dtype="float32")
    mat[0][0] = 1.0 / (aspect * tan_half_fovy)
    mat[1][1] = 1.0 / tan_half_fovy
    mat[2][2] = z_far / (z_near + z_far)
    mat[2][3] = 1.0
    mat[3][2] = -(z_far * z_near) / (z_far - z_near)
    return mat


def look_at(eye, center, pose, xp=None):
    xp = np if xp is None else xp
    f = normalize(center - eye)
    s = normalize(xp.cross(f, pose))
    u = xp.cross(s, f)
    mat = xp.eye(4, dtype="float32")
    mat[0][0] = s[0]
    mat[1][0] = s[1]
    mat[2][0] = s[2]
    mat[0][1] = u[0]
    mat[1][1] = u[1]
    mat[2][1] = u[2]
    mat[0][2] = -f[0]
    mat[1][2] = -f[1]
    mat[2][2] = -f[2]
    mat[3][0] = -xp.dot(s, eye)
    mat[3][1] = -xp.dot(u, eye)
    mat[3][2] = xp.dot(f, eye)
    return mat
