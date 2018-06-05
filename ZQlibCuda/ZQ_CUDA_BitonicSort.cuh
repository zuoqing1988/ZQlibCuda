#ifndef _ZQ_CUDA_BITONIC_SORT_H_
#define _ZQ_CUDA_BITONIC_SORT_H_


#include "ZQlibCudaDefines.cuh"
#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_BitonicSort
{
	bool cu_BitonicSort(float* data, int len, bool ascending_dir);
}

#endif