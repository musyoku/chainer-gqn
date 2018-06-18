#pragma once
#include "../renderer/glm.h"
#include <memory>
#include <pybind11/numpy.h>

namespace three {
namespace base {
    namespace py = pybind11;
    class Object {
    public:
        void update_model_matrix();
        std::unique_ptr<glm::vec3i[]> _faces;
        std::unique_ptr<glm::vec3f[]> _vertices;
        std::unique_ptr<glm::vec3f[]> _face_vertices;
        std::unique_ptr<glm::vec3f[]> _face_normal_vectors;
        std::unique_ptr<glm::vec3f[]> _face_vertex_normal_vectors;
        std::unique_ptr<glm::vec4f[]> _face_vertex_colors;
        int _num_faces;
        int _num_vertices;
        glm::vec3 _position; // xyz
        glm::vec3 _rotation_rad; // xyz
        glm::vec3 _scale; // xyz
        glm::mat4 _model_matrix;
        void reserve(int num_faces, int num_vertices);
        void set_scale(py::tuple scale);
        void set_position(py::tuple position);
        void set_rotation(py::tuple rotation_rad);
    };
}
}