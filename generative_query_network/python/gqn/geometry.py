import os
import numpy as np


def load_vertices(filepath):
    vertices = []
    with open(filepath) as file:
        for line in file:
            vertex = [float(v) for v in line.split()]
            vertex.append(1.0) # 同次座標系
            vertices.append(vertex)
    return np.vstack(vertices).astype(np.float32)


def load_faces(filepath):
    faces = []
    with open(filepath) as file:
        for line in file:
            faces.append([float(f) for f in line.split()])
    return np.vstack(faces).astype(np.int32)


def load(directory):
    vertices = load_vertices(os.path.join(directory, "vertices"))
    faces = load_faces(os.path.join(directory, "faces"))
    return vertices, faces


def save_vertices(filepath, vertices):
    with open(filepath, "w") as file:
        lines = []
        for vertex in vertices:
            lines.append("{} {} {}".format(vertex[0], vertex[1], vertex[2]))
        file.write("\n".join(lines))


def save_faces(filepath, faces):
    with open(filepath, "w") as file:
        lines = []
        for face in faces:
            lines.append("{} {} {}".format(face[0], face[1], face[2]))
        file.write("\n".join(lines))


def save(directory, vertices, faces):
    try:
        os.mkdir(directory)
    except:
        pass
    save_vertices(os.path.join(directory, "vertices"), vertices)
    save_faces(os.path.join(directory, "faces"), faces)