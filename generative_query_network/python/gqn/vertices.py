import math, chainer
from . import matrix


def deg2rad(angle_degree):
    return angle_degree % 360 / 180.0 * math.pi


def rotate_x(vertices, angle_degree):
    xp = chainer.cuda.get_array_module(vertices)
    rad = deg2rad(angle_degree)
    rotation_mat = matrix.rotation_x(rad, xp)
    vertices = xp.dot(vertices, rotation_mat.T)
    return vertices.astype(xp.float32)


def rotate_y(vertices, angle_degree):
    xp = chainer.cuda.get_array_module(vertices)
    rad = deg2rad(angle_degree)
    rotation_mat = matrix.rotation_y(rad, xp)
    vertices = xp.dot(vertices, rotation_mat.T)
    return vertices.astype(xp.float32)


def rotate_z(vertices, angle_degree):
    xp = chainer.cuda.get_array_module(vertices)
    rad = deg2rad(angle_degree)
    rotation_mat = matrix.rotation_z(rad, xp)
    vertices = xp.dot(vertices, rotation_mat.T)
    return vertices.astype(xp.float32)


# 各面の各頂点番号に対応する座標を取る
def convert_to_face_representation(vertices, faces):
    assert (vertices.ndim == 2)
    assert (faces.ndim == 2)
    assert (vertices.shape[1] == 4)
    assert (faces.shape[1] == 3)

    xp = chainer.cuda.get_array_module(faces)
    num_vertices = vertices.shape[0]
    vertices = vertices.reshape((num_vertices, 4)).astype(xp.float32)
    return vertices[faces]