#ifndef _ZQ_CUDA_WAVELET_H_
#define _ZQ_CUDA_WAVELET_H_

namespace ZQ_CUDA_Wavelet
{
	/*db1 filter, zero padding mode*/
	extern "C"
	float DWT2(const float* input, const int width, const int height, const int nLevels, float*& output, int& output_width, int& output_height);
}

#endif