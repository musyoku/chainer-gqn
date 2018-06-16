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
                glm::mat4 pvm_mat = projection_mat * view_mat * model_mat;
                update_depth_map(object_index, object.get(), pvm_mat, np_face_index_map, np_depth_map);
            }
        }

        // 各画素ごとに最前面を特定する
        void update_depth_map(
            int object_index,
            scene::Object* object,
            glm::mat4 pvm_mat,
            py::array_t<int, py::array::c_style>& np_face_index_map,
            py::array_t<float, py::array::c_style>& np_depth_map)
        {
            std::unique_ptr<glm::vec3i[]>& faces = object->_faces;
            std::unique_ptr<glm::vec4f[]>& vertices = object->_vertices;

            int num_faces = object->_num_faces;
            int map_height = np_depth_map.shape(0);
            int map_width = np_depth_map.shape(1);

            auto depth_map = np_depth_map.mutable_unchecked<2>();
            auto object_index_map = np_face_index_map.mutable_unchecked<2>();
            auto face_index_map = np_face_index_map.mutable_unchecked<2>();

            auto to_projected_coordinate = [](int p, int size) -> float {
                return 2.0 * (p / (float)(size - 1) - 0.5);
            };

            for (int face_index = 0; face_index < num_faces; face_index++) {
                glm::vec3i& face = faces[face_index];
                glm::vec4f vfa = pvm_mat * vertices[face[0]];
                glm::vec4f vfb = pvm_mat * vertices[face[1]];
                glm::vec4f vfc = pvm_mat * vertices[face[2]];

                // 同次座標のwの絶対値で割ると可視領域のz座標の範囲が[0, 1]になる
                vfa /= glm::abs(vfa.w);
                vfb /= glm::abs(vfb.w);
                vfc /= glm::abs(vfc.w);

                std::cout << "a: " << vfa.x << ", " << vfa.y << ", " << vfa.z << std::endl;
                std::cout << "b: " << vfb.x << ", " << vfb.y << ", " << vfb.z << std::endl;
                std::cout << "c: " << vfc.x << ", " << vfc.y << ", " << vfc.z << std::endl;

                // カリングによる裏面のスキップ
                // 面の頂点の並び（1 -> 2 -> 3）が時計回りの場合描画しない
                if ((vfa.y - vfc.y) * (vfa.x - vfb.x) < (vfa.y - vfb.y) * (vfa.x - vfc.x)) {
                    continue;
                }

                // 全画素についてループ
                for (int yi = 0; yi < map_height; yi++) {
                    // yi \in [0, map_height] -> yf \in [-1, 1]
                    float yf = -to_projected_coordinate(yi, map_height);
                    // y座標が面の外部ならスキップ
                    if ((yf > vfa.y && yf > vfb.y && yf > vfc.y) || (yf < vfa.y && yf < vfb.y && yf < vfc.y)) {
                        continue;
                    }
                    for (int xi = 0; xi < map_width; xi++) {
                        // xi \in [0, map_width] -> xf \in [-1, 1]
                        float xf = to_projected_coordinate(xi, map_width);

                        // xyが面の外部ならスキップ
                        // Edge Functionで3辺のいずれかの右側にあればスキップ
                        // https://www.cs.drexel.edu/~david/Classes/Papers/comp175-06-pineda.pdf
                        if ((yf - vfa.y) * (vfb.x - vfa.x) < (xf - vfa.x) * (vfb.y - vfa.y) || (yf - vfb.y) * (vfc.x - vfb.x) < (xf - vfb.x) * (vfc.y - vfb.y) || (yf - vfc.y) * (vfa.x - vfc.x) < (xf - vfc.x) * (vfa.y - vfc.y)) {
                            continue;
                        }

                        // 重心座標系の各係数を計算
                        // http://zellij.hatenablog.com/entry/20131207/p1
                        float lambda_1 = ((vfb.y - vfc.y) * (xf - vfc.x) + (vfc.x - vfb.x) * (yf - vfc.y)) / ((vfb.y - vfc.y) * (vfa.x - vfc.x) + (vfc.x - vfb.x) * (vfa.y - vfc.y));
                        float lambda_2 = ((vfc.y - vfa.y) * (xf - vfc.x) + (vfa.x - vfc.x) * (yf - vfc.y)) / ((vfb.y - vfc.y) * (vfa.x - vfc.x) + (vfc.x - vfb.x) * (vfa.y - vfc.y));
                        float lambda_3 = 1.0 - lambda_1 - lambda_2;

                        // 面f_nのxy座標に対応する点のz座標を求める
                        // https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
                        float z_face = 1.0 / (lambda_1 / vfa.z + lambda_2 / vfb.z + lambda_3 / vfc.z);

                        if (z_face < 0.0 || z_face > 1.0) {
                            continue;
                        }
                        // zは小さい方が手前
                        float current_min_z = depth_map(yi, xi);
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