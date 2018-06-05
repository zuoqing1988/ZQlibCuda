#ifndef _ZQ_CUDA_IMAGE_PROCESSING_2D_CUH_
#define _ZQ_CUDA_IMAGE_PROCESSING_2D_CUH_

#include "ZQlibCudaDefines.cuh"
#include <stdio.h>
#include <stdlib.h>


namespace ZQ_CUDA_ImageProcessing2D
{
	
	/********************************************************/
	void cu_WarpImage_Bilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels);
	
	void cu_WarpImage_Bicubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels);

	void cu_ResizeImage_Bilinear(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels);

	void cu_ResizeImage_Bicubic(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels);

	void cu_AddWith(float* in_out_put, const float* other, const int width, const int height, const int nChannels);
	
	void cu_MulWithScale(float* in_out_put, const float scale, const int width, const int height, const int nChannels);

	void cu_Add_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float* weight2, 
											const int width, const int height, const int nChannels);
	void cu_GaussianSmoothing(float* dst, const float* src, const int width, const int height, const int nChannels);

	void cu_GaussianSmoothing2(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int nChannels);

	void cu_DerivativeX(float* dst, const float* src, const int width, const int height, const int nChannels);

	void cu_DerivativeY(float* dst, const float* src, const int width, const int height, const int nChannels);

	void cu_DxForward(float* dst, const float* src, const int width, const int height, const int nChannels);

	void cu_DyForward(float* dst, const float* src, const int width, const int height, const int nChannels);

	void cu_Laplacian(float* dst, const float* src, const int width, const int height, const int nChannels);

	void cu_CopyChannel_i(float* dst, const float* src, const int i, const int width, const int height, const int nChannels);
	
	
	/**************************   Kernel functions    **********************************/
	//assume Im2 is binded to tex_img_1channel
	__global__
	void WarpImage_1channel_Kernel(float* warpIm2, const float* Im1, const float* u, const float* v, const int width, const int height);
	
	//assume Im2 is binded to tex_img_4channel
	__global__
	void WarpImage_4channel_Kernel(float4* warpIm2, const float4* Im1, const float* u, const float* v, const int width, const int height);

	__global__
	void WarpImage_Bilinear_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels); 
	
	__global__
	void WarpImage_Bilinear_Occupy_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels); 

	__global__
	void WarpImage_Bicubic_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels);
	
	__global__
	void WarpImage_Bicubic_Occupy_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels);

	//assume Src is binded to tex_img_1channel
	__global__
	void ResizeImage_1channel_Kernel(float* dst, const int dst_width, const int dst_height);

	//assume Src is binded to tex_img_4channel
	__global__
	void ResizeImage_4channel_Kernel(float4* dst, const int dst_width, const int dst_height);

	__global__
	void ResizeImage_Bilinear_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels);

	__global__
	void ResizeImage_Bicubic_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels);
	
	__global__
	void Addwith_Kernel(float* in_out_put, const float* other, const float weight, const int width, const int height, const int nChannels);
	
	__global__
	void Add_Im1_weight1_Im2_weight2_Kernel(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2, 
											const int width, const int height, const int nChannels);											

	__global__
	void Imfilter_h_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);
	
	__global__
	void Imfilter_v_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);
	
	__global__
	void Imfilter_h_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int nChannels);
	
	__global__
	void Imfilter_v_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int nChannels);
	
	__global__
	void Derivative_x_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);
	
	__global__
	void Derivative_y_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);

	__global__
	void Dx_Forward_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);

	__global__
	void Dy_Forward_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);
	
	__global__
	void Laplacian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels);

	__global__
	void CopyChannel_i_Kernel(float* output, const float* input, const int i, const int width, const int height, const int nChannels);

	__global__
	void MedianFilterWithMask5x5_Kernel(float* output, const float* input, const int width, const int height, const int nChannels, const bool* keep_mask);
	
	__global__
	void MedianFilterWithMask3x3_Kernel(float* output, const float* input, const int width, const int height, const int nChannels, const bool* keep_mask);
}

#endif