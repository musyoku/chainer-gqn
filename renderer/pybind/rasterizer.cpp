#include "../core/rasterizer.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(rasterizer_cpu, module)
{
    module.def("update_depth_map", &gqn::rasterizer::update_depth_map);
}