#include "scene.h"

namespace three {
namespace scene {
    Scene::Scene()
    {
    }
    void Scene::add(Object* object)
    {
        _objects.emplace_back(object);
    }
}
}