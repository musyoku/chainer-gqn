#pragma once
#include "../../scene/scene.h"
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <memory>
#include <pybind11/numpy.h>

namespace three {
namespace renderer {
    namespace opengl {
        namespace py = pybind11;
        class VertexArrayObject {
        private:
            const GLuint _attribute_position = 0;
            const GLuint _attribute_face_normal_vector = 1;
            const GLuint _attribute_vertex_normal_vector = 2;
            const GLuint _attribute_vertex_color = 3;
            std::unique_ptr<GLuint[]> _vao;
            std::unique_ptr<GLuint[]> _vbo_vertices;
            std::unique_ptr<GLuint[]> _vbo_normal_vectors;
            std::unique_ptr<GLuint[]> _vbo_vertex_normal_vectors;
            std::unique_ptr<GLuint[]> _vbo_vertex_colors;
            std::unique_ptr<GLuint[]> _vbo_faces;
            int _prev_num_objects;
            void delete_buffers();

        public:
            ~VertexArrayObject();
            void build(scene::Scene* scene);
            void bind_object(int object_index);
        };
    }
}
}