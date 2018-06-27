#include "vao.h"

namespace three {
namespace renderer {
    namespace opengl {
        VertexArrayObject::~VertexArrayObject()
        {
            delete_buffers();
        }
        void VertexArrayObject::delete_buffers()
        {
            if (_prev_num_objects == -1) {
                return;
            }
            glDeleteVertexArrays(_prev_num_objects, _vao.get());
            glDeleteBuffers(_prev_num_objects, _vbo_faces.get());
            glDeleteBuffers(_prev_num_objects, _vbo_vertices.get());
            glDeleteBuffers(_prev_num_objects, _vbo_normal_vectors.get());
            glDeleteBuffers(_prev_num_objects, _vbo_vertex_normal_vectors.get());
            glDeleteBuffers(_prev_num_objects, _vbo_vertex_colors.get());
        }
        void VertexArrayObject::build(scene::Scene* scene)
        {
            delete_buffers();

            int num_objects = scene->_objects.size();
            _prev_num_objects = num_objects;

            _vao = std::make_unique<GLuint[]>(num_objects);
            glGenVertexArrays(num_objects, _vao.get());

            _vbo_faces = std::make_unique<GLuint[]>(num_objects);
            glGenBuffers(num_objects, _vbo_faces.get());

            _vbo_vertices = std::make_unique<GLuint[]>(num_objects);
            glGenBuffers(num_objects, _vbo_vertices.get());

            _vbo_normal_vectors = std::make_unique<GLuint[]>(num_objects);
            glGenBuffers(num_objects, _vbo_normal_vectors.get());

            _vbo_vertex_normal_vectors = std::make_unique<GLuint[]>(num_objects);
            glGenBuffers(num_objects, _vbo_vertex_normal_vectors.get());

            _vbo_vertex_colors = std::make_unique<GLuint[]>(num_objects);
            glGenBuffers(num_objects, _vbo_vertex_colors.get());

            for (int n = 0; n < num_objects; n++) {
                glBindVertexArray(_vao[n]);

                auto& object = scene->_objects[n];
                int num_faces = object->_num_faces;
                int num_vertices = object->_num_vertices;

                glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertices[n]);
                glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_vertices.get(), GL_STATIC_DRAW);
                glVertexAttribPointer(_attribute_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
                glEnableVertexAttribArray(_attribute_position);

                glBindBuffer(GL_ARRAY_BUFFER, _vbo_normal_vectors[n]);
                glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_normal_vectors.get(), GL_STATIC_DRAW);
                glVertexAttribPointer(_attribute_face_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
                glEnableVertexAttribArray(_attribute_face_normal_vector);

                glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertex_normal_vectors[n]);
                glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_vertex_normal_vectors.get(), GL_STATIC_DRAW);
                glVertexAttribPointer(_attribute_vertex_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
                glEnableVertexAttribArray(_attribute_vertex_normal_vector);

                glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertex_colors[n]);
                glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec4f), object->_face_vertex_colors.get(), GL_STATIC_DRAW);
                glVertexAttribPointer(_attribute_vertex_color, 4, GL_FLOAT, GL_FALSE, 0, 0);
                glEnableVertexAttribArray(_attribute_vertex_color);

                glBindVertexArray(0);
            }

            glBindVertexArray(0);
        }
        void VertexArrayObject::bind_object(int object_index)
        {
            if (object_index >= _prev_num_objects) {
                throw std::runtime_error("(object_index >= _prev_num_objects) -> false");
            }
            glBindVertexArray(_vao[object_index]);
        }
    }
}
}