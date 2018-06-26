#include "cornell_box.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>

namespace three {
namespace scene {
    CornellBox::CornellBox(
        py::tuple north_wall_color,
        py::tuple east_wall_color,
        py::tuple south_wall_color,
        py::tuple west_wall_color,
        py::tuple scale)
    {
        _num_faces = 12;
        _num_vertices = 8;
        _smoothness = false;
        reserve(_num_faces, _num_vertices);

        _faces[0] = glm::vec3i(0, 1, 2);
        _faces[1] = glm::vec3i(0, 2, 3);
        _faces[2] = glm::vec3i(0, 4, 1);
        _faces[3] = glm::vec3i(1, 4, 5);
        _faces[4] = glm::vec3i(1, 5, 6);
        _faces[5] = glm::vec3i(1, 6, 2);
        _faces[6] = glm::vec3i(2, 6, 3);
        _faces[7] = glm::vec3i(3, 6, 7);
        _faces[8] = glm::vec3i(3, 7, 0);
        _faces[9] = glm::vec3i(0, 7, 4);
        _faces[10] = glm::vec3i(4, 7, 6);
        _faces[11] = glm::vec3i(4, 6, 5);

        _vertices[0] = glm::vec3f(0.5, 0.5, 0.5);
        _vertices[1] = glm::vec3f(-0.5, 0.5, 0.5);
        _vertices[2] = glm::vec3f(-0.5, 0.5, -0.5);
        _vertices[3] = glm::vec3f(0.5, 0.5, -0.5);
        _vertices[4] = glm::vec3f(0.5, -0.5, 0.5);
        _vertices[5] = glm::vec3f(-0.5, -0.5, 0.5);
        _vertices[6] = glm::vec3f(-0.5, -0.5, -0.5);
        _vertices[7] = glm::vec3f(0.5, -0.5, -0.5);

        std::unique_ptr<glm::vec3f[]> vertex_normal_vectors = std::make_unique<glm::vec3f[]>(_num_vertices);
        for (ssize_t vertex_index = 0; vertex_index < _num_vertices; vertex_index++) {
            vertex_normal_vectors[vertex_index] = glm::vec3f(0.0);
        }

        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            glm::vec3i face = _faces[face_index];

            glm::vec3f va = _vertices[face[0]];
            glm::vec3f vb = _vertices[face[1]];
            glm::vec3f vc = _vertices[face[2]];

            glm::vec3f vba = vb - va;
            glm::vec3f vca = vc - va;
            glm::vec3f normal = glm::normalize(glm::cross(vba, vca));

            _face_vertices[face_index * 3 + 0] = va;
            _face_vertices[face_index * 3 + 1] = vb;
            _face_vertices[face_index * 3 + 2] = vc;

            _face_normal_vectors[face_index * 3 + 0] = normal;
            _face_normal_vectors[face_index * 3 + 1] = normal;
            _face_normal_vectors[face_index * 3 + 2] = normal;

            vertex_normal_vectors[face[0]] += normal;
            vertex_normal_vectors[face[1]] += normal;
            vertex_normal_vectors[face[2]] += normal;
        }

        for (ssize_t vertex_index = 0; vertex_index < _num_vertices; vertex_index++) {
            vertex_normal_vectors[vertex_index] = glm::normalize(vertex_normal_vectors[vertex_index]);
        }

        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            glm::vec3i face = _faces[face_index];
            _face_vertex_normal_vectors[face_index * 3 + 0] = vertex_normal_vectors[face[0]];
            _face_vertex_normal_vectors[face_index * 3 + 1] = vertex_normal_vectors[face[1]];
            _face_vertex_normal_vectors[face_index * 3 + 2] = vertex_normal_vectors[face[2]];
        }

        _position = glm::vec3(0.0);
        _rotation_rad = glm::vec3(0.0);

        glm::vec4f white(1.0, 1.0, 1.0, 1.0);
        _face_vertex_colors[0 * 3 + 0] = white;
        _face_vertex_colors[0 * 3 + 1] = white;
        _face_vertex_colors[0 * 3 + 2] = white;
        _face_vertex_colors[1 * 3 + 0] = white;
        _face_vertex_colors[1 * 3 + 1] = white;
        _face_vertex_colors[1 * 3 + 2] = white;
        _face_vertex_colors[10 * 3 + 0] = white;
        _face_vertex_colors[10 * 3 + 1] = white;
        _face_vertex_colors[10 * 3 + 2] = white;
        _face_vertex_colors[11 * 3 + 0] = white;
        _face_vertex_colors[11 * 3 + 1] = white;
        _face_vertex_colors[11 * 3 + 2] = white;

        set_north_wall_color(north_wall_color);
        set_east_wall_color(east_wall_color);
        set_south_wall_color(south_wall_color);
        set_west_wall_color(west_wall_color);

        set_scale(scale);
        update_model_matrix();
    }
    CornellBox::CornellBox(const CornellBox* source)
    {
        _num_faces = source->_num_faces;
        _num_vertices = source->_num_vertices;
        _smoothness = source->_smoothness;
        reserve(_num_faces, _num_vertices);

        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            _faces[face_index] = source->_faces[face_index];
            for (int offset = 0; offset < 3; offset++) {
                _face_vertices[face_index * 3 + offset] = source->_face_vertices[face_index * 3 + offset];
                _face_vertex_colors[face_index * 3 + offset] = source->_face_vertex_colors[face_index * 3 + offset];
                _face_normal_vectors[face_index * 3 + offset] = source->_face_normal_vectors[face_index * 3 + offset];
                _face_vertex_normal_vectors[face_index * 3 + offset] = source->_face_vertex_normal_vectors[face_index * 3 + offset];
            }
        }

        for (ssize_t vertex_index = 0; vertex_index < _num_vertices; vertex_index++) {
            _vertices[vertex_index] = source->_vertices[vertex_index];
        }

        _position = source->_position;
        _rotation_rad = source->_rotation_rad;
        _scale = source->_scale;
        _model_matrix = source->_model_matrix;
        _north_wall_color = source->_north_wall_color;
        _east_wall_color = source->_east_wall_color;
        _south_wall_color = source->_south_wall_color;
        _west_wall_color = source->_west_wall_color;
    }
    void CornellBox::set_north_wall_color(py::tuple north_wall_color)
    {
        _north_wall_color[0] = north_wall_color[0].cast<float>();
        _north_wall_color[1] = north_wall_color[1].cast<float>();
        _north_wall_color[2] = north_wall_color[2].cast<float>();
        _north_wall_color[3] = north_wall_color[3].cast<float>();

        _face_vertex_colors[2 * 3 + 0] = _north_wall_color;
        _face_vertex_colors[2 * 3 + 1] = _north_wall_color;
        _face_vertex_colors[2 * 3 + 2] = _north_wall_color;

        _face_vertex_colors[3 * 3 + 0] = _north_wall_color;
        _face_vertex_colors[3 * 3 + 1] = _north_wall_color;
        _face_vertex_colors[3 * 3 + 2] = _north_wall_color;
    }
    void CornellBox::set_east_wall_color(py::tuple east_wall_color)
    {
        _east_wall_color[0] = east_wall_color[0].cast<float>();
        _east_wall_color[1] = east_wall_color[1].cast<float>();
        _east_wall_color[2] = east_wall_color[2].cast<float>();
        _east_wall_color[3] = east_wall_color[3].cast<float>();

        _face_vertex_colors[4 * 3 + 0] = _east_wall_color;
        _face_vertex_colors[4 * 3 + 1] = _east_wall_color;
        _face_vertex_colors[4 * 3 + 2] = _east_wall_color;

        _face_vertex_colors[5 * 3 + 0] = _east_wall_color;
        _face_vertex_colors[5 * 3 + 1] = _east_wall_color;
        _face_vertex_colors[5 * 3 + 2] = _east_wall_color;
    }
    void CornellBox::set_south_wall_color(py::tuple south_wall_color)
    {
        _south_wall_color[0] = south_wall_color[0].cast<float>();
        _south_wall_color[1] = south_wall_color[1].cast<float>();
        _south_wall_color[2] = south_wall_color[2].cast<float>();
        _south_wall_color[3] = south_wall_color[3].cast<float>();

        _face_vertex_colors[6 * 3 + 0] = _south_wall_color;
        _face_vertex_colors[6 * 3 + 1] = _south_wall_color;
        _face_vertex_colors[6 * 3 + 2] = _south_wall_color;

        _face_vertex_colors[7 * 3 + 0] = _south_wall_color;
        _face_vertex_colors[7 * 3 + 1] = _south_wall_color;
        _face_vertex_colors[7 * 3 + 2] = _south_wall_color;
    }
    void CornellBox::set_west_wall_color(py::tuple west_wall_color)
    {
        _west_wall_color[0] = west_wall_color[0].cast<float>();
        _west_wall_color[1] = west_wall_color[1].cast<float>();
        _west_wall_color[2] = west_wall_color[2].cast<float>();
        _west_wall_color[3] = west_wall_color[3].cast<float>();

        _face_vertex_colors[8 * 3 + 0] = _west_wall_color;
        _face_vertex_colors[8 * 3 + 1] = _west_wall_color;
        _face_vertex_colors[8 * 3 + 2] = _west_wall_color;

        _face_vertex_colors[9 * 3 + 0] = _west_wall_color;
        _face_vertex_colors[9 * 3 + 1] = _west_wall_color;
        _face_vertex_colors[9 * 3 + 2] = _west_wall_color;
    }
    std::shared_ptr<CornellBox> CornellBox::clone()
    {
        return std::make_shared<CornellBox>(this);
    }
}
}