#include "pytorch_cpp_helper.hpp"

int FRForwardLauncher(
    const Tensor features,
    const Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    Tensor output);

int FRBackwardLauncher(
    const Tensor top_grad,
    const Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    Tensor bottom_grad);

int feature_refine_forward(
    const Tensor features, 
    const Tensor best_bboxes,                       
    const float spatial_scale,
    const int points,
    Tensor output) {

    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(best_bboxes);
    CHECK_CUDA_INPUT(output);

    return FRForwardLauncher(
        features,
        best_bboxes,
        spatial_scale, 
        points,
        output
    );
}

int feature_refine_backward(
    const Tensor top_grad,
    const Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    Tensor bottom_grad) {

    CHECK_CUDA_INPUT(top_grad);
    CHECK_CUDA_INPUT(best_bboxes);
    CHECK_CUDA_INPUT(bottom_grad);

    return FRBackwardLauncher(
        top_grad, 
        best_bboxes,
        spatial_scale,
        points,
        bottom_grad
    );
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", &feature_refine_forward, "Feature Refine forward (CUDA)");
//     m.def("backward", &feature_refine_backward, "Feature Refine backward (CUDA)");
// }
