#ifndef _ZQ_CUDA_IMAGE_PROCESSING_2D_H_
#define _ZQ_CUDA_IMAGE_PROCESSING_2D_H_

namespace ZQ_CUDA_ImageProcessing2D
{
	/*************************  extern "C"  functions *********************************/
	extern "C" float WarpImage2D_Bilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels);

	extern "C" float WarpImage2D_Bilinear_Occupy(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels);

	extern "C" float WarpImage2D_Bicubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels);

	extern "C" float WarpImage2D_Bicubic_Occupy(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels);

	extern "C" float ResizeImage2D_Bilinear(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels);

	extern "C" float ResizeImage2D_Bicubic(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels);

	extern "C" float AddWith2D(float* in_out_put, const float* other, const int width, const int height, const int nChannels);

	extern "C" float MulWithScale2D(float* in_out_put, const float scale, const int width, const int height, const int nChannels);

	extern "C" float Add2D_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float* weight2,
		const int width, const int height, const int nChannels);

	extern "C" float GaussianSmoothing2D(float* dst, const float* src, const int width, const int height, const int nChannels);
	
	extern "C" float GaussianSmoothing2_2D(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int nChannels);

	extern "C" float DerivativeX2D(float* dst, const float* src, const int width, const int height, const int nChannels);

	extern "C" float DerivativeY2D(float* dst, const float* src, const int width, const int height, const int nChannels);

	extern "C" float DxForward2D(float* dst, const float* src, const int width, const int height, const int nChannels);

	extern "C" float DyForward2D(float* dst, const float* src, const int width, const int height, const int nChannels);

	extern "C" float Laplacian2D(float* dst, const float* src, const int width, const int height, const int nChannels);

}
#endif