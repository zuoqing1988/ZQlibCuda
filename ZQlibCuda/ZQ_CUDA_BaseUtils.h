#ifndef _ZQ_CUDA_BASE_UTILS_H_
#define _ZQ_CUDA_BASE_UTILS_H_

namespace ZQ_CUDA_BaseUtils
{
	extern "C"
		bool Find_MAX_Value(const int num, const float* vals, float& result);

	extern "C"
		bool Find_MIN_Value(const int num, const float* vals, float& result);

}

#endif