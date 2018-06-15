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
    assert (vertices.ndim == 3)
    assert (faces.ndim == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 4)
    assert (faces.shape[2] == 3)

    xp = chainer.cuda.get_array_module(faces)
    batch_size, num_vertices = vertices.shape[:2]
    faces = faces + (
        xp.arange(batch_size, dtype=xp.int32) * num_vertices)[:, None, None]
    vertices = vertices.reshape((batch_size * num_vertices,
                                 4)).astype(xp.float32)
    return vertices[faces]