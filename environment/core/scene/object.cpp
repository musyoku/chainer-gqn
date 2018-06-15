#include "object.h"
#include <stdexcept>

namespace environment {
namespace scene {
    Object::Object(pybind11::array_t<int> np_faces, pybind11::array_t<float> np_vertices)
    {
        if (np_faces.ndim() != 2) {
            throw std::runtime_error("(np_faces.ndim() != 2) -> false");
        }
        if (np_vertices.ndim() != 2) {
            throw std::runtime_error("(np_vertices.ndim() != 2) -> false");
        }
        if (np_faces.shape(1) != 3) {
            throw std::runtime_error("(np_faces.shape(1) != 3) -> false");
        }
        if (np_vertices.shape(1) != 4) {
            throw std::runtime_error("(np_vertices.shape(1) != 4) -> false");
        }
        _num_faces = np_faces.shape(0);
        _num_vertices = np_vertices.shape(0);
        _faces = std::make_unique<int[]>(_num_faces * 3);
        _vertices = std::make_unique<float[]>(_num_vertices * 4);

        auto faces = np_faces.mutable_unchecked<2>();
        for (ssize_t face_index = 0; face_index < np_faces.shape(0); face_index++) {
            int index = face_index * 3;
            _faces[index + 0] = faces(face_index, 0);
            _faces[index + 1] = faces(face_index, 1);
            _faces[index + 2] = faces(face_index, 2);
        }

        auto vertices = np_vertices.mutable_unchecked<2>();
        for (ssize_t vertex_index = 0; vertex_index < np_vertices.shape(0); vertex_index++) {
            int index = vertex_index * 4;
            _vertices[index + 0] = vertices(vertex_index, 0); // x
            _vertices[index + 1] = vertices(vertex_index, 1); // y
            _vertices[index + 2] = vertices(vertex_index, 2); // z
            _vertices[index + 4] = vertices(vertex_index, 4); // w
        }
    }
}
}