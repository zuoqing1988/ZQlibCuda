#ifndef _ZQ_CUDA_MATRIX_MUL_CUH_
#define _ZQ_CUDA_MATRIX_MUL_CUH_

#include "ZQlibCudaDefines.cuh"

#include <stdio.h>
#include <stdlib.h>



namespace ZQ_CUDA_MatrixMul
{
	/**************************** CUDA functions Begin ***********************/
	void cu_MatrixMul_BlockSize(const float* A, const float* B, const int hA, const int wA, const int wB, float* C);

	/**************************** Kernel functions Begin **************************/
	__global__ void ZQ_Cuda_MatrixMul_Kernel(const float* d_A, const float* d_B, const int wA, const int wB, float* d_C);
}

#endif