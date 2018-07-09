#pragma once
#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

namespace three {
namespace base {
    namespace py = pybind11;
    class Light {
    public:
        glm::vec3 _position;
        float _intensity;
        Light(float intensity);
        Light(py::tuple position, float intensity);
        void set_position(py::tuple position);
        void set_intensity(float intensity);
    };
}
}