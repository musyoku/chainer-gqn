#include "rasterizer.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace gqn {
namespace rasterizer {
    float to_projected_coordinate(int p, int size)
    {
        return 2.0 * (p / (float)(size - 1) - 0.5);
    }

    // 各画素ごとに最前面を特定する
    void update_depth_map(
        py::array_t<float, py::array::c_style> np_face_vertices,
        py::array_t<int, py::array::c_style> np_face_index_map,
        py::array_t<float, py::array::c_style> np_depth_map)
    {
        if (np_face_vertices.ndim() != 3) {
            throw std::runtime_error("(np_face_vertices.ndim() != 3) -> true");
        }
        if (np_depth_map.ndim() != 2) {
            throw std::runtime_error("(np_depth_map.ndim() != 2) -> true");
        }
        if (np_face_index_map.ndim() != 2) {
            throw std::runtime_error("(np_face_index_map.ndim() != 2) -> true");
        }

        int num_faces = np_face_vertices.shape(0);
        int image_height = np_depth_map.shape(0);
        int image_width = np_depth_map.shape(1);

        auto face_vertices = np_face_vertices.mutable_unchecked<3>();
        auto depth_map = np_depth_map.mutable_unchecked<2>();
        auto face_index_map = np_face_index_map.mutable_unchecked<2>();

        for (int face_index = 0; face_index < num_faces; face_index++) {
            float xf_1 = face_vertices(face_index, 0, 0);
            float yf_1 = face_vertices(face_index, 0, 1);
            float zf_1 = face_vertices(face_index, 0, 2);
            float xf_2 = face_vertices(face_index, 1, 0);
            float yf_2 = face_vertices(face_index, 1, 1);
            float zf_2 = face_vertices(face_index, 1, 2);
            float xf_3 = face_vertices(face_index, 2, 0);
            float yf_3 = face_vertices(face_index, 2, 1);
            float zf_3 = face_vertices(face_index, 2, 2);

            // カリングによる裏面のスキップ
            // 面の頂点の並び（1 -> 2 -> 3）が時計回りの場合描画しない
            if ((yf_1 - yf_3) * (xf_1 - xf_2) < (yf_1 - yf_2) * (xf_1 - xf_3)) {
                continue;
            }

            // 全画素についてループ
            for (int yi = 0; yi < image_height; yi++) {
                // yi \in [0, image_height] -> yf \in [-1, 1]
                float yf = -to_projected_coordinate(yi, image_height);
                // y座標が面の外部ならスキップ
                if ((yf > yf_1 && yf > yf_2 && yf > yf_3) || (yf < yf_1 && yf < yf_2 && yf < yf_3)) {
                    continue;
                }
                for (int xi = 0; xi < image_width; xi++) {
                    // xi \in [0, image_width] -> xf \in [-1, 1]
                    float xf = to_projected_coordinate(xi, image_width);

                    // xyが面の外部ならスキップ
                    // Edge Functionで3辺のいずれかの右側にあればスキップ
                    // https://www.cs.drexel.edu/~david/Classes/Papers/comp175-06-pineda.pdf
                    if ((yf - yf_1) * (xf_2 - xf_1) < (xf - xf_1) * (yf_2 - yf_1) || (yf - yf_2) * (xf_3 - xf_2) < (xf - xf_2) * (yf_3 - yf_2) || (yf - yf_3) * (xf_1 - xf_3) < (xf - xf_3) * (yf_1 - yf_3)) {
                        continue;
                    }

                    // 重心座標系の各係数を計算
                    // http://zellij.hatenablog.com/entry/20131207/p1
                    float lambda_1 = ((yf_2 - yf_3) * (xf - xf_3) + (xf_3 - xf_2) * (yf - yf_3)) / ((yf_2 - yf_3) * (xf_1 - xf_3) + (xf_3 - xf_2) * (yf_1 - yf_3));
                    float lambda_2 = ((yf_3 - yf_1) * (xf - xf_3) + (xf_1 - xf_3) * (yf - yf_3)) / ((yf_2 - yf_3) * (xf_1 - xf_3) + (xf_3 - xf_2) * (yf_1 - yf_3));
                    float lambda_3 = 1.0 - lambda_1 - lambda_2;

                    // 面f_nのxy座標に対応する点のz座標を求める
                    // https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
                    float z_face = 1.0 / (lambda_1 / zf_1 + lambda_2 / zf_2 + lambda_3 / zf_3);

                    if (z_face < 0.0 || z_face > 1.0) {
                        continue;
                    }
                    // zは小さい方が手前
                    float current_min_z = depth_map(yi, xi);
                    if (z_face < current_min_z) {
                        // 現在の面の方が前面の場合
                        depth_map(yi, xi) = z_face;
                        face_index_map(yi, xi) = face_index;
                    }
                }
            }
        }
    }
}
}