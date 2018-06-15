#include "scene.h"

namespace environment {
namespace scene {
    void Scene::add(Object* object)
    {
        _objects.emplace_back(object);
    }
}
}