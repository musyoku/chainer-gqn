#pragma once
#include "../base/object.h"
#include "../renderer/glm.h"
#include <memory>
#include <pybind11/numpy.h>

namespace three {
namespace scene {
    namespace py = pybind11;
    class CornellBox : public base::Object {
    public:
        glm::vec4 _north_wall_color; // RGBA
        glm::vec4 _east_wall_color; // RGBA
        glm::vec4 _south_wall_color; // RGBA
        glm::vec4 _west_wall_color; // RGBA
        CornellBox(const CornellBox* source);
        CornellBox(
            py::tuple north_wall_color,
            py::tuple east_wall_color,
            py::tuple south_wall_color,
            py::tuple west_wall_color,
            py::tuple scale);
        void set_north_wall_color(py::tuple north_wall_color);
        void set_east_wall_color(py::tuple east_wall_color);
        void set_south_wall_color(py::tuple south_wall_color);
        void set_west_wall_color(py::tuple west_wall_color);
        std::shared_ptr<CornellBox> clone();
    };
}
}