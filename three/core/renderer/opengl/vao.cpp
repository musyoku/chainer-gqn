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
            glCreateVertexArrays(num_objects, _vao.get());

            _vbo_faces = std::make_unique<GLuint[]>(num_objects);
            glCreateBuffers(num_objects, _vbo_faces.get());

            _vbo_vertices = std::make_unique<GLuint[]>(num_objects);
            glCreateBuffers(num_objects, _vbo_vertices.get());

            _vbo_normal_vectors = std::make_unique<GLuint[]>(num_objects);
            glCreateBuffers(num_objects, _vbo_normal_vectors.get());

            _vbo_vertex_normal_vectors = std::make_unique<GLuint[]>(num_objects);
            glCreateBuffers(num_objects, _vbo_vertex_normal_vectors.get());

            _vbo_vertex_colors = std::make_unique<GLuint[]>(num_objects);
            glCreateBuffers(num_objects, _vbo_vertex_colors.get());

            for (int object_index = 0; object_index < num_objects; object_index++) {
                GLuint vao = _vao[object_index];
                auto& object = scene->_objects[object_index];
                int num_faces = object->_num_faces;
                int num_vertices = object->_num_vertices;

                glNamedBufferData(_vbo_vertices[object_index], 3 * num_faces * sizeof(glm::vec3f), object->_face_vertices.get(), GL_STATIC_DRAW);
                glEnableVertexArrayAttrib(vao, _attribute_position);
                glVertexArrayAttribFormat(vao, _attribute_position, 3, GL_FLOAT, GL_FALSE, 0);
                glVertexArrayAttribBinding(vao, _attribute_position, 0);
                glVertexArrayVertexBuffer(vao, 0, _vbo_vertices[object_index], 0, sizeof(glm::vec3f));

                glNamedBufferData(_vbo_normal_vectors[object_index], 3 * num_faces * sizeof(glm::vec3f), object->_face_normal_vectors.get(), GL_STATIC_DRAW);
                glEnableVertexArrayAttrib(vao, _attribute_face_normal_vector);
                glVertexArrayAttribFormat(vao, _attribute_face_normal_vector, 3, GL_FLOAT, GL_FALSE, 0);
                glVertexArrayAttribBinding(vao, _attribute_face_normal_vector, 1);
                glVertexArrayVertexBuffer(vao, 1, _vbo_normal_vectors[object_index], 0, sizeof(glm::vec3f));

                glNamedBufferData(_vbo_vertex_normal_vectors[object_index], 3 * num_faces * sizeof(glm::vec3f), object->_face_vertex_normal_vectors.get(), GL_STATIC_DRAW);
                glEnableVertexArrayAttrib(vao, _attribute_vertex_normal_vector);
                glVertexArrayAttribFormat(vao, _attribute_vertex_normal_vector, 3, GL_FLOAT, GL_FALSE, 0);
                glVertexArrayAttribBinding(vao, _attribute_vertex_normal_vector, 2);
                glVertexArrayVertexBuffer(vao, 2, _vbo_vertex_normal_vectors[object_index], 0, sizeof(glm::vec3f));

                glNamedBufferData(_vbo_vertex_colors[object_index], 3 * num_faces * sizeof(glm::vec4f), object->_face_vertex_colors.get(), GL_STATIC_DRAW);
                glEnableVertexArrayAttrib(vao, _attribute_vertex_color);
                glVertexArrayAttribFormat(vao, _attribute_vertex_color, 3, GL_FLOAT, GL_FALSE, 0);
                glVertexArrayAttribBinding(vao, _attribute_vertex_color, 3);
                glVertexArrayVertexBuffer(vao, 3, _vbo_vertex_colors[object_index], 0, sizeof(glm::vec4f));
            }
        }
        void VertexArrayObject::bind_object(int object_index)
        {
            if (object_index >= _prev_num_objects) {
                throw std::runtime_error("(object_index >= _prev_num_objects) -> false");
            }
            glBindVertexArray(_vao[object_index]);
            // glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertices[object_index]);
            // glVertexAttribPointer(_attribute_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
            // glEnableVertexAttribArray(_attribute_position);
            // glVertexAttribPointer(_attribute_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
            // glEnableVertexAttribArray(_attribute_position);
            // glVertexAttribPointer(_attribute_face_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            // glEnableVertexAttribArray(_attribute_face_normal_vector);
            // glVertexAttribPointer(_attribute_vertex_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            // glEnableVertexAttribArray(_attribute_vertex_normal_vector);
            // glVertexAttribPointer(_attribute_vertex_color, 4, GL_FLOAT, GL_FALSE, 0, 0);
            // glEnableVertexAttribArray(_attribute_vertex_color);
        }
    }
}
}