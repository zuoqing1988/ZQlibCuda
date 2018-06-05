#ifndef _ZQ_CUDA_IMAGE_PROCESSING_3D_CUH_
#define _ZQ_CUDA_IMAGE_PROCESSING_3D_CUH_

#include "ZQlibCudaDefines.cuh"
#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_ImageProcessing3D
{
//	extern texture<float,3,cudaReadModeElementType> tex_img_1channel;
//	extern texture<float4,3,cudaReadModeElementType> tex_img_4channel;
	
	/*************************  extern "C"  functions *********************************/
//	extern "C" float WarpImage_Trilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w, 
//							const int width, const int height, const int depth, const int nChannels);
							
//	extern "C" float WarpImage_Tricubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w,
//							const int width, const int height, const int depth, const int nChannels);
							
//	extern "C" float ResizeImage_Trilinear(float* dst, const float* src, const int src_width, const int src_height, const int src_depth, 
//							const int dst_width, const int dst_height, const int dst_depth, const int nChannels);
							
//	extern "C" float ResizeImage_Tricubic(float* dst, const float* src, const int src_width, const int src_height, const float src_depth, 
//							const int dst_width, const int dst_height, const int dst_depth, const int nChannels);
							
//	extern "C" float Addwith(float* in_out_put, const float* other, const float weight, const int width, const int height, const int depth, const int nChannels);
	
//	extern "C" float Add_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2,
//									const int width, const int height, const int depth, const int nChannels);
									
//	extern "C" float GausssianSmoothing(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);
	
//	extern "C" float GaussianSmoothing2(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int depth, const int nChannels);
	
//	extern "C" float DerivativeX(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);
	
//	extern "C" float DerivativeY(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);
	
//	extern "C" float DerivativeZ(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);
	
//	extern "C" float Laplacian(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels);
	
	
	
	/**************************   Kernel functions    **********************************/
	//assume Im2 is binded to tex_img_1channel
	__global__
	void WarpImage_1channel_Kernel(float* warpIm2, const float* Im1, const float* u, const float* v, const float* w, const int width, const int height, const int depth);
	
	//assume Im2 is binded to tex_img_4channel
	__global__
	void WarpImage_4channel_Kernel(float4* warpIm2, const float4* Im1, const float* u, const float* v, const float* w, const int width, const int height, const int depth);

	__global__
	void WarpImage_Trilinear_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w, 
										const int width, const int height, const int depth, const int nChannels); 

	__global__
	void WarpImage_Tricubic_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w, 
							const int width, const int height, const int depth, const int nChannels);
	
	//assume Src is binded to tex_img_1channel
	__global__
	void ResizeImage_1channel_Kernel(float* dst, const int dst_width, const int dst_height, const int dst_depth);

	//assume Src is binded to tex_img_4channel
	__global__
	void ResizeImage_4channel_Kernel(float4* dst, const int dst_width, const int dst_height, const int dst_depth);

	__global__
	void ResizeImage_Trilinear_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int src_depth,
					const int dst_width, const int dst_height, const int dst_depth, const int nChannels);
	__global__
	void ResizeImage_Tricubic_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int src_depth, 
								const int dst_width, const int dst_height, const int dst_depth, const int nChannels);
								
	__global__
	void Addwith_Kernel(float* in_out_put, const float* other, const float weight, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Add_Im1_weight1_Im2_weight2_Kernel(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2, 
											const int width, const int height, const int depth, const int nChannels);										

	__global__
	void Imfilter_h_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Imfilter_v_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Imfilter_d_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Imfilter_h_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Imfilter_v_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Imfilter_d_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Derivative_x_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Derivative_y_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Derivative_z_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void Laplacian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels);
	
}

#endif