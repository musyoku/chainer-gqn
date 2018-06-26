#pragma once
#include "../base/object.h"
#include "../renderer/glm.h"
#include <memory>
#include <pybind11/numpy.h>

namespace three {
namespace scene {
    namespace py = pybind11;
    class Object : public base::Object {
    public:
        glm::vec4 _color; // RGBA
        Object(const Object* source);
        Object(py::array_t<int> np_faces,
            py::array_t<float> np_vertices,
            py::tuple color,
            py::tuple scale,
            bool smoothness);
        void set_color(py::tuple color);
        std::shared_ptr<Object> clone();
    };
}
}