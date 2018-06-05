#ifndef _ZQ_CUDA_BLEND_TWO_IMAGES3D_CUH_
#define _ZQ_CUDA_BLEND_TWO_IMAGES3D_CUH_

#include "ZQlibCudaDefines.cuh"

#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_BlendTwoImages3D
{
	void cu_ScatteredInterpolation(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const int nChannels, const float* values, const float radius, const int iterations,
								   const int out_width, const int out_height, const int out_depth, float* out_images, bool various_neighbor_num);
								   
	void cu_BlendTwoImages(const int width, const int height, const int depth, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float* w,
								const float weight1, float* out_image, bool various_neighbor_num);
	
}

#endif