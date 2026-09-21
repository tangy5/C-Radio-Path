// Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.

#include <string>

#include <torch/extension.h>

torch::Tensor spatialtransform_cpu(torch::Tensor inputs, torch::Tensor stms, int output_width,
                                   int output_height, std::string method, float background,
                                   bool verbose);
