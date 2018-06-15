#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <pybind11/numpy.h>

namespace glm {
typedef vec<4, float> vec4f;
typedef vec<3, int> vec3i;
}

namespace environment {
namespace scene {
    namespace py = pybind11;
    class Object {
    public:
        std::unique_ptr<glm::vec3i[]> _faces;
        std::unique_ptr<glm::vec4f[]> _vertices;
        int _num_faces;
        int _num_vertices;
        glm::vec3 _location; // xyz
        glm::vec3 _rotation_rad; // xyz
        glm::vec4 _color; // RGBA
        glm::mat4 _model_matrix;
        Object(py::array_t<int> np_faces, py::array_t<float> np_vertices, py::tuple location, py::tuple rotation_rad, py::tuple color);

    };
}
}