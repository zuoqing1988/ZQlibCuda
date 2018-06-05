#ifndef _ZQ_CUDA_MUL_H_
#define _ZQ_CUDA_MUL_H_

namespace ZQ_CUDA_MatrixMul
{
	extern "C" float MatrixMul(const float* A, const float* B, const int hA, const int wA, const int wB, float* C);
}


#endif