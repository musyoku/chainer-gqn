#include "scene.h"

namespace three {
namespace scene {
    Scene::Scene()
    {
    }
    void Scene::add(std::shared_ptr<Object> object)
    {
        _objects.push_back(object);
    }
}
}