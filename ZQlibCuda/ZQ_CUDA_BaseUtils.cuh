#ifndef _ZQ_CUDA_BASE_UTILS_CUH_
#define _ZQ_CUDA_BASE_UTILS_CUH_

#include "ZQlibCudaDefines.cuh"
#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_BaseUtils
{
	/*** cuda functions ***/
	bool cu_Find_MAX_Value(const int num, const float* vals, float& result);
	bool cu_Find_MAX_Value(const int num, const int* vals, int& result);
	
	bool cu_Find_MIN_Value(const int num, const float* vals, float& result);
	bool cu_Find_MIN_Value(const int num, const int* vals, int& result);
	
	bool cu_SUM(const int num, const float* vals, float& result);
	bool cu_SUM(const int num, const int* vals, int& result);
	
	bool cu_SUM_Square(const int num, const float* vals, float& result);
	bool cu_SUM_Square(const int num, const int* vals, int& result);
	
	void cu_Square(const int num, const float* input, float* output);
	void cu_Square(const int num, const int* input, int* output);
	
	void cu_Abs(const int num, const float* input, float* output);
	void cu_Abs(const int num, const int* input, int* output);
	
	void cu_Compute_bucket_stored_offset(const int bucket_num, const int* bucket_stored_num, int* bucket_stored_offset);
	
	void cu_Distribute_Bucket(const int num, const int bucket_num, int* bucket_stored_num, int* bucket_stored_offset, int* bucket_stored_index, const int* coord_in_which_bucket);
}

#endif