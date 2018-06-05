#ifndef _ZQ_CUDA_WAVELET_CUH_
#define _ZQ_CUDA_WAVELET_CUH_

#include "ZQlibCudaDefines.cuh"
#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_Wavelet
{
	/* make sure [width, height, levels] are compatible*/
	void cu_DWT2_NLevels(const float* input, const int width, const int height, const int levels, float* output);
	
	void cu_DWT2_NLevels(const float* input, const int width, const int height, const int depth, const int levels, float* output);
}

#endif