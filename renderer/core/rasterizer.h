#pragma once
#include <pybind11/numpy.h>

namespace gqn {
namespace rasterizer {
    namespace py = pybind11;
    void update_depth_map(
        py::array_t<float, py::array::c_style> np_face_vertices,
        py::array_t<int, py::array::c_style> np_face_index_map,
        py::array_t<float, py::array::c_style> np_depth_map);
}
}