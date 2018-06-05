#ifndef _ZQ_CUDA_BLEND_TWO_IMAGES_CUH_
#define _ZQ_CUDA_BLEND_TWO_IMAGES_CUH_

#include "ZQlibCudaDefines.cuh"
#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_BlendTwoImages
{
	void cu_ScatteredInterpolation(const int num, const float* coord_x, const float* coord_y, const int nChannels, const float* values, const float radius, const int iterations,
								   const int out_width, const int out_height, float* out_images, bool various_neighbor_num);
					
	/*sample_mode:	0: bilinear;	1: bicubic;		2: interger (i.e. nearest point)
	blend_mode:		0: both			1: current frame;	2: next frame;
	*/
	void cu_BlendTwoImages(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float weight1,
		const int skip, const float radius, const int iterations, float* out_image, const bool various_neighbor_num, const int sample_mode, const int blend_mode);
	
	/*sample_mode:	0: bilinear;	1: bicubic;		2: interger (i.e. nearest point)
	blend_mode:		0: both			1: current frame;	2: next frame;		
	*/
	void cu_BlendTwoImagesByMedFilt(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float weight1,
		float* out_image, const int sample_mode, const int blend_mode);

	void cu_InterpolateVelocityByMedFilt_4channels(const int width, const int height, const float* u, const float* v, const float weight1, float* vel_image);
}

#endif