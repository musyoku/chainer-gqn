#pragma once
#include "../pass.h"
#include <gl3w/gl3w.h>
#include <memory>

namespace three {
namespace renderer {
    namespace multipass {
        class ScreenSpaceAmbientOcculusion : public RenderPass {
        private:
            GLuint _sampling_points_sampler_id;
            GLuint _sampling_points_texture_id;
            GLuint _depth_buffer_sampler_id;
            GLuint _render_result_texture_id;
            std::unique_ptr<GLfloat[]> _sampling_points_data;
            int _viewport_width;
            int _viewport_height;
            int _total_sampling_points;

        public:
            ScreenSpaceAmbientOcculusion(int viewport_width, int viewport_height, int total_sampling_points);
            bool bind(int num_sampling_points, float intensity, GLuint depth_buffer_texture_id);
            bool bind(GLuint depth_buffer_texture_id);
            void unbind();
            GLuint get_buffer_texture_id();
        };
    }
}
}