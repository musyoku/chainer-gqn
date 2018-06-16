#pragma once
#include "object.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

namespace three {
namespace scene {
    namespace py = pybind11;
    class Scene {
    public:
        std::vector<std::shared_ptr<Object>> _objects;
        Scene();
        void add(std::shared_ptr<Object> object);
        void add(std::shared_ptr<Object> object, py::tuple position);
        void add(std::shared_ptr<Object> object, py::tuple position, py::tuple rotation_rad);
    };
}
}