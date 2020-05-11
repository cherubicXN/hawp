#pragma once
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> lsencode_cuda(
    const at::Tensor& lines,
    const int input_height,
    const int input_width,
    const int height,
    const int width,
    const int num_lines);