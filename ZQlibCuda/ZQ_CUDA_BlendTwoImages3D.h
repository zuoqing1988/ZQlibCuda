#ifndef _ZQ_CUDA_BLEND_TWO_IMAGES3D_H_
#define _ZQ_CUDA_BLEND_TWO_IMAGES3D_H_

namespace ZQ_CUDA_BlendTwoImages3D
{
	extern "C"
	float Cutil_ScatteredInterpolation3D(const int num, const float* coord_x, const float* coord_y, const float* coord_z, 
			const int nChannels, const float* values, const float radius, const int iterations,
			const int out_width, const int out_height, const int out_depth, float* out_images, bool various_neighbor_num);

	extern "C"
	float Cutil_BlendTwoImages3D(const int width, const int height, const int depth, const int nChannels, const float* image1, const float* image2,
			const float* u, const float* v, const float* w,
			const float weight1, const int skip, const float radius, const int iterations, float* out_image, bool various_neighbor_num, bool cubic);

}

#endif