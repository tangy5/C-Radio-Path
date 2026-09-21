/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Perspective warp using 3x3 homography matrices.
 * Dispatches to CPU (OpenMP) or CUDA based on input tensor device.
 * Ported from EVFM (libs/pytorch_image_ops/spatial_transform/).
 */

#include "spatial_transform.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_transform", &spatialtransform, "Perspective warp (CPU/CUDA)",
          py::arg("inputs"), py::arg("stms"), py::arg("output_width"),
          py::arg("output_height"), py::arg("method"), py::arg("background"),
          py::arg("verbose"));
}
