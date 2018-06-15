#pragma once
#include "object.h"
#include <memory>
#include <vector>

namespace three {
namespace scene {
    class Scene {
    public:
        std::vector<Object*> _objects;
        Scene();
        void add(Object* object);
    };
}
}