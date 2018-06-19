#include "object.h"
#include "../opengl.h"
#include <glm/glm.hpp>
#include <iostream>

namespace imgplot {
namespace view {
    ObjectView::ObjectView(data::ObjectData* data, double x, double y, double width, double height)
        : View(x, y, width, height)
    {
        _data = data;
        _renderer = std::make_unique<renderer::ObjectRenderer>(data->extracted_vertices(), data->num_extracted_vertices(), data->faces(), data->num_faces());
    }
    void ObjectView::_bind_vertices()
    {
        _renderer->update_vertices(_data->extracted_vertices(), _data->num_extracted_vertices());
    }
    void ObjectView::_bind_faces()
    {
        _renderer->update_faces(_data->faces(), _data->num_faces());
    }
    void ObjectView::_bind_normal_vectors()
    {
        _renderer->update_normal_vectors(_data->normal_vectors(), _data->num_extracted_vertices());
    }
    void ObjectView::render(double aspect_ratio)
    {
        if (_data->vertices_updated()) {
            _bind_vertices();
        }
        if (_data->faces_updated()) {
            _bind_faces();
        }
        if (_data->normal_vector_updated()) {
            _bind_normal_vectors();
        }
        _renderer->render(aspect_ratio);
    }
    void ObjectView::zoom_in()
    {
        _renderer->zoom_in();
    }
    void ObjectView::zoom_out()
    {
        _renderer->zoom_out();
    }
    void ObjectView::rotate_camera(double diff_x, double diff_y)
    {
        _renderer->rotate_camera(diff_x, diff_y);
    }
}
}
