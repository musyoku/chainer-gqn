#include "scene.h"
#include <iostream>

namespace three {
namespace scene {
    Scene::Scene()
    {
    }
    void Scene::add(std::shared_ptr<Object> object)
    {
        _objects.push_back(object);
    }
    void Scene::add(std::shared_ptr<Object> object, py::tuple position)
    {
        object->set_position(position);
        _objects.push_back(object);
    }
    void Scene::add(std::shared_ptr<Object> object, py::tuple position, py::tuple rotation_rad)
    {
        object->set_position(position);
        object->set_rotation(rotation_rad);
        _objects.push_back(object);
    }
    void Scene::add(std::shared_ptr<CornellBox> object)
    {
        _objects.push_back(object);
    }
    void Scene::add(std::shared_ptr<CornellBox> object, py::tuple position)
    {

        object->set_position(position);
        _objects.push_back(object);
    }
    void Scene::add(std::shared_ptr<CornellBox> object, py::tuple position, py::tuple rotation_rad)
    {
        object->set_position(position);
        object->set_rotation(rotation_rad);
        _objects.push_back(object);
    }
}
}