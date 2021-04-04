#include "pytorch_cpp_helper.hpp"
#include <cmath>
#include <vector>

int ROIAlignRotatedForwardLaucher(const at::Tensor features, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor output);

int ROIAlignRotatedBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                                   const float spatial_scale, const int sample_num,
                                   const int channels, const int height,
                                   const int width, const int num_rois,
                                   const int pooled_height, const int pooled_width,
                                   at::Tensor bottom_grad);

int roi_align_rotated_forward_cuda(at::Tensor features, at::Tensor rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, int sample_num,
                           at::Tensor output) {
  CHECK_CUDA_INPUT(features);
  CHECK_CUDA_INPUT(rois);
  CHECK_CUDA_INPUT(output);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ROIAlignRotatedForwardLaucher(features, rois, spatial_scale, sample_num,
                         num_channels, data_height, data_width, num_rois,
                         pooled_height, pooled_width, output);

  return 1;
}

int roi_align_rotated_backward_cuda(at::Tensor top_grad, at::Tensor rois,
                            int pooled_height, int pooled_width,
                            float spatial_scale, int sample_num,
                            at::Tensor bottom_grad) {
  CHECK_CUDA_INPUT(top_grad);
  CHECK_CUDA_INPUT(rois);
  CHECK_CUDA_INPUT(bottom_grad);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 6) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = bottom_grad.size(1);
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);

  ROIAlignRotatedBackwardLaucher(top_grad, rois, spatial_scale, sample_num,
                          num_channels, data_height, data_width, num_rois,
                          pooled_height, pooled_width, bottom_grad);

  return 1;
}
