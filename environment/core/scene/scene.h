#pragma once
#include "object.h"
#include <memory>
#include <vector>

namespace environment {
namespace scene {
    class Scene {
    public:
        std::vector<Object*> _objects;
        void add(Object* object);
    };
}
}