#include "../core/renderer/depth_map.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace three;

PYBIND11_MODULE(renderer, module)
{
    module.def("render_depth_map", &renderer::rasterizer::render_depth_map);
}