from . import rasterizer_cpu


def update_depth_map(face_vertices, face_index_map, depth_map):
    rasterizer_cpu.update_depth_map(face_vertices, face_index_map, depth_map)
