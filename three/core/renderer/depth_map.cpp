#include "depth_map.h"
#include "../scene/object.h"
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

namespace three {
namespace renderer {
    namespace rasterizer {
        void render_depth_map(scene::Scene* scene, camera::PerspectiveCamera* camera,
            py::array_t<int, py::array::c_style> np_face_index_map,
            py::array_t<int, py::array::c_style> np_object_index_map,
            py::array_t<float, py::array::c_style> np_depth_map)
        {
            if (np_depth_map.ndim() != 2) {
                throw std::runtime_error("(np_depth_map.ndim() != 2) -> true");
            }
            if (np_face_index_map.ndim() != 2) {
                throw std::runtime_error("(np_face_index_map.ndim() != 2) -> true");
            }
            if (np_object_index_map.ndim() != 2) {
                throw std::runtime_error("(np_object_index_map.ndim() != 2) -> true");
            }
            glm::mat4& view_mat = camera->_view_matrix;
            glm::mat4& projection_mat = camera->_projection_matrix;
            std::vector<std::shared_ptr<scene::Object>>& objects = scene->_objects;
            for (int object_index = 0; object_index < objects.size(); object_index++) {
                std::shared_ptr<scene::Object> object = objects[object_index];
                glm::mat4& model_mat = object->_model_matrix;
                update_depth_map(object_index, object.get(), model_mat, view_mat, projection_mat, np_face_index_map, np_depth_map);
            }
        }
        // 各画素ごとに最前面を特定する
        void update_depth_map(
            int object_index,
            scene::Object* object,
            glm::mat4 model_mat,
            glm::mat4 view_mat,
            glm::mat4 projection_mat,
            py::array_t<int, py::array::c_style>& np_face_index_map,
            py::array_t<float, py::array::c_style>& np_depth_map)
        {
            std::unique_ptr<glm::vec3i[]>& faces = object->_faces;
            std::unique_ptr<glm::vec4f[]>& vertices = object->_vertices;

            int num_faces = object->_num_faces;
            int map_height = np_depth_map.shape(0);
            int map_width = np_depth_map.shape(1);

            glm::mat4 camera_mat = view_mat * model_mat;
            glm::mat4 pvm_mat = projection_mat * camera_mat;

            // std::cout << map_width << "x" << map_height << std::endl;

            auto depth_map = np_depth_map.mutable_unchecked<2>();
            auto object_index_map = np_face_index_map.mutable_unchecked<2>();
            auto face_index_map = np_face_index_map.mutable_unchecked<2>();

            auto to_projected_coordinate = [](int p, int size) -> double {
                return 2.0 * ((2.0 * p + 1.0) / (double)(2.0 * size) - 0.5);
            };

            for (int face_index = 0; face_index < num_faces; face_index++) {
                // if (!(face_index == 8 || face_index == 9)) {
                //     continue;
                // }
                glm::vec3i& face = faces[face_index];
                glm::vec4f cvfa = camera_mat * vertices[face[0]];
                glm::vec4f cvfb = camera_mat * vertices[face[1]];
                glm::vec4f cvfc = camera_mat * vertices[face[2]];

                glm::vec4f vfa = projection_mat * cvfa;
                glm::vec4f vfb = projection_mat * cvfb;
                glm::vec4f vfc = projection_mat * cvfc;

                // 同次座標のwの絶対値で割ると可視領域のz座標の範囲が[0, 1]になる
                glm::vec4f nvfa = vfa / glm::abs(vfa.w);
                glm::vec4f nvfb = vfb / glm::abs(vfb.w);
                glm::vec4f nvfc = vfc / glm::abs(vfc.w);

                // std::cout << "face: " << face_index << std::endl;
                // std::cout << "a: " << nvfa.x << ", " << nvfa.y << ", " << nvfa.z << std::endl;
                // std::cout << "b: " << nvfb.x << ", " << nvfb.y << ", " << nvfb.z << std::endl;
                // std::cout << "c: " << nvfc.x << ", " << nvfc.y << ", " << nvfc.z << std::endl;

                // カリングによる裏面のスキップ
                // 面の頂点の並び（1 -> 2 -> 3）が時計回りの場合描画しない
                if ((nvfa.y - nvfc.y) * (nvfa.x - nvfb.x) < (nvfa.y - nvfb.y) * (nvfa.x - nvfc.x)) {
                    // std::cout << "skipped" << std::endl;
                    continue;
                }

                // 全画素についてループ
                for (int yi = 0; yi < map_height; yi++) {
                    // yi \in [0, map_height] -> yf \in [-1, 1]
                    double yf = -to_projected_coordinate(yi, map_height);
                    // std::cout << "yi = " << yi << ", yf = " << yf << std::endl;
                    // y座標が面の外部ならスキップ
                    if ((yf > nvfa.y && yf > nvfb.y && yf > nvfc.y) || (yf < nvfa.y && yf < nvfb.y && yf < nvfc.y)) {
                        // std::cout << "outside y: " << yi << " : " << yf << std::endl;
                        continue;
                    }
                    for (int xi = 0; xi < map_width; xi++) {

                        // std::cout << "face: " << face_index << std::endl;
                        // std::cout << "a: " << nvfa.x << ", " << nvfa.y << ", " << nvfa.z << std::endl;
                        // std::cout << "b: " << nvfb.x << ", " << nvfb.y << ", " << nvfb.z << std::endl;
                        // std::cout << "c: " << nvfc.x << ", " << nvfc.y << ", " << nvfc.z << std::endl;

                        // xi \in [0, map_width] -> xf \in [-1, 1]
                        double xf = to_projected_coordinate(xi, map_width);
                        // std::cout << "xi = " << xi << ", xf = " << xf << std::endl;

                        // xyが面の外部ならスキップ
                        // Edge Functionで3辺のいずれかの右側にあればスキップ
                        // https://www.cs.drexel.edu/~david/Classes/Papers/comp175-06-pineda.pdf
                        if ((yf - nvfa.y) * (nvfb.x - nvfa.x) < (xf - nvfa.x) * (nvfb.y - nvfa.y) || (yf - nvfb.y) * (nvfc.x - nvfb.x) < (xf - nvfb.x) * (nvfc.y - nvfb.y) || (yf - nvfc.y) * (nvfa.x - nvfc.x) < (xf - nvfc.x) * (nvfa.y - nvfc.y)) {
                            // std::cout << "outside face: " << xi << ", " << yi << " : " << xf << ", " << yf << std::endl;
                            continue;
                        }

                        // 重心座標系の各係数を計算
                        // http://zellij.hatenablog.com/entry/20131207/p1
                        double lambda_1 = ((nvfb.y - nvfc.y) * (xf - nvfc.x) + (nvfc.x - nvfb.x) * (yf - nvfc.y)) / ((nvfb.y - nvfc.y) * (nvfa.x - nvfc.x) + (nvfc.x - nvfb.x) * (nvfa.y - nvfc.y));
                        double lambda_2 = ((nvfc.y - nvfa.y) * (xf - nvfc.x) + (nvfa.x - nvfc.x) * (yf - nvfc.y)) / ((nvfb.y - nvfc.y) * (nvfa.x - nvfc.x) + (nvfc.x - nvfb.x) * (nvfa.y - nvfc.y));
                        double lambda_3 = 1.0 - lambda_1 - lambda_2;

                        // 面f_nのxy座標に対応する点のz座標を求める
                        // https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
                        double z_face = 1.0 / (lambda_1 / cvfa.z + lambda_2 / cvfb.z + lambda_3 / cvfc.z);
                        // double z_face = lambda_1 * nvfa.z + lambda_2 * nvfb.z + lambda_3 * nvfc.z;

                        // ###################################
                        // double x_face = lambda_1 * nvfa.x + lambda_2 * nvfb.x + lambda_3 * nvfc.x;
                        // double y_face = lambda_1 * nvfa.y + lambda_2 * nvfb.y + lambda_3 * nvfc.y;

                        // if(glm::abs(x_face - xf) > 1e-12){
                        //     std::cout << x_face << " == " << xf;
                        //     throw std::runtime_error("bug");
                        // }
                        // if(glm::abs(y_face - yf) > 1e-12){
                        //     std::cout << y_face << " == " << yf;
                        //     throw std::runtime_error("bug");
                        // }

                        // std::cout << "lambda: " << lambda_1 << ", " << lambda_2 << ", " << lambda_3 << std::endl;
                        // std::cout << x_face << " == " << xf << std::endl;
                        // std::cout << y_face << " == " << yf << std::endl;

                        double z_min = std::min(std::min(cvfa.z, cvfb.z), cvfc.z);
                        double z_max = std::max(std::max(cvfa.z, cvfb.z), cvfc.z);
                        if (z_face < z_min || z_max < z_face) {
                            std::cout << z_min << " < " << z_face << " < " << z_max << std::endl;
                            std::cout << "face: " << face_index << std::endl;
                            std::cout << "a: " << cvfa.x << ", " << cvfa.y << ", " << cvfa.z << ", " << cvfa.w << std::endl;
                            std::cout << "b: " << cvfb.x << ", " << cvfb.y << ", " << cvfb.z << ", " << cvfb.w << std::endl;
                            std::cout << "c: " << cvfc.x << ", " << cvfc.y << ", " << cvfc.z << ", " << cvfc.w << std::endl;
                            std::cout << "lambda: " << lambda_1 << ", " << lambda_2 << ", " << lambda_3 << std::endl;
                            throw std::runtime_error("bug");
                        }
                        // ###################################

                        if (z_face < 0.0 || z_face > 1.0) {
                            // std::cout << "z_face outside: " << z_face << std::endl;
                            continue;
                        }
                        // zは小さい方が手前
                        double current_min_z = depth_map(yi, xi);
                        // std::cout << current_min_z << " < " << z_face << std::endl;
                        if (z_face < current_min_z) {
                            // 現在の面の方が前面の場合
                            depth_map(yi, xi) = z_face;
                            face_index_map(yi, xi) = face_index;
                            object_index_map(yi, xi) = object_index;
                        }
                    }
                }
            }
        }
    }
}
}