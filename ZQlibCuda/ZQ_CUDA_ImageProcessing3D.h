#ifndef _ZQ_CUDA_IMAGE_PROCESSING_3D_H_
#define _ZQ_CUDA_IMAGE_PROCESSING_3D_H_

namespace ZQ_CUDA_ImageProcessing3D
{
	extern "C" 
	float WarpImage3D_Trilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w,
						const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float WarpImage3D_Tricubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w,
						const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float ResizeImage3D_Trilinear(float* dst, const float* src, const int src_width, const int src_height, const int src_depth,
						const int dst_width, const int dst_height, const int dst_depth, const int nChannels);

	extern "C" 
	float ResizeImage3D_Tricubic(float* dst, const float* src, const int src_width, const int src_height, const float src_depth,
						const int dst_width, const int dst_height, const int dst_depth, const int nChannels);

	extern "C" 
	float Addwith3D(float* in_out_put, const float* other, const float weight, const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float Add3D_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2,
						const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float GausssianSmoothing3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float GaussianSmoothing2_3D(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float DerivativeX3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float DerivativeY3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float DerivativeZ3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);

	extern "C" 
	float Laplacian3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);
}

#endif