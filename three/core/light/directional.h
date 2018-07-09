#pragma once
#include "../base/light.h"
#include <pybind11/pybind11.h>

namespace three {
namespace light {
    namespace py = pybind11;
    class DirectionalLight : public base::Light {
    };
}
}