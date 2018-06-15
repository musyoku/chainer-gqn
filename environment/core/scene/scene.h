#pragma once
#include "object.h"
#include <memory>
#include <vector>

namespace environment {
namespace scene {
    class Scene {
    public:
        std::vector<std::unique_ptr<Object>> _objects;
    };
}
}