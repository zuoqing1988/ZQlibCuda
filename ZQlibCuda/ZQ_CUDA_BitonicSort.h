#ifndef _ZQ_CUDA_BITONIC_SORT_H_
#define _ZQ_CUDA_BITONIC_SORT_H_

namespace ZQ_CUDA_BitonicSort
{
	extern "C"
	bool BitonicSort(float* data, int len, bool ascending_dir, float& cost_time);
}

#endif