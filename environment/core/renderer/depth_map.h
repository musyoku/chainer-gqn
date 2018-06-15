#pragma once
#include "../camera/perspective.h"
#include "../scene/scene.h"
#include <pybind11/numpy.h>

namespace environment {
namespace renderer {
    namespace rasterizer {
        namespace py = pybind11;
        void render_depth_map(
            scene::Scene* scene, camera::PerspectiveCamera* camera,
            py::array_t<int, py::array::c_style> np_face_index_map,
            py::array_t<int, py::array::c_style> np_object_index_map,
            py::array_t<float, py::array::c_style> np_depth_map);
        void update_depth_map(
            int object_index,
            scene::Object* object,
            py::array_t<int, py::array::c_style>& np_face_index_map,
            py::array_t<float, py::array::c_style>& np_depth_map);
    }
}
}