#ifndef _ZQ_CUDA_COMPRESS_IMAGE_H_
#define _ZQ_CUDA_COMPRESS_IMAGE_H_

namespace ZQ_CUDA_CompressImage
{
	extern "C"
	float ComputeCoeffsAndThresh(const float* input, const int width, const int height, const int nLevels, const float quality, float*& output, int& out_width, int& out_height, float& thresh);

	extern "C"
	float ComputeCoeffsAndThresh3D(const float* input, const int width, const int height, const int depth, const int nLevels, const float quality, float*& output, int& out_width, int& out_height, float& thresh);
}

#endif