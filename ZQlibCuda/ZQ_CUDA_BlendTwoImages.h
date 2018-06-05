#ifndef _ZQ_CUDA_BLEND_TWO_IMAGES_H_
#define _ZQ_CUDA_BLEND_TWO_IMAGES_H_

namespace ZQ_CUDA_BlendTwoImages
{
	extern "C"
	float Cutil_ScatteredInterpolation(const int num, const float* coord_x, const float* coord_y, const int nChannels, const float* values, const float radius, const int iterations,
		const int out_width, const int out_height, float* out_images, bool various_neighbor_num);


	/*sample_mode:	0: bilinear;	1: bicubic;		2: interger (i.e. nearest point)
	blend_mode:		0: both			1: current frame;	2: next frame;
	*/
	extern "C"
	float Cutil_BlendTwoImages(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, 
		const float weight1, const int skip, const float radius, const int iterations, float* out_image, const bool various_neighbor_num, const int sample_mode, const int blend_mode = 0);


	/*sample_mode:	0: bilinear;	1: bicubic;		2: interger (i.e. nearest point)
	blend_mode:		0: both			1: current frame;	2: next frame;
	*/
	extern "C"
	float Cutil_BlendTwoImagesByMedFilt(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float*v,
		const float weight1, float* out_image, const int sample_mode, const int blend_mode = 0);
}

#endif