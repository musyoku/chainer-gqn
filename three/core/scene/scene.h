#pragma once
#include "object.h"
#include <memory>
#include <vector>

namespace three {
namespace scene {
    class Scene {
    public:
        std::vector<std::shared_ptr<Object>> _objects;
        Scene();
        void add(std::shared_ptr<Object> object);
    };
}
}