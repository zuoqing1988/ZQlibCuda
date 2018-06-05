#ifndef _ZQ_CUDA_MERGE_DENSITY_2D_UTILS_CUH_
#define _ZQ_CUDA_MERGE_DENSITY_2D_UTILS_CUH_


#include "ZQlibCudaDefines.cuh"
#include <vector_types.h>


namespace ZQ_CUDA_MergeDensity2D
{
	/****************   Base Kernels  **********************************/
	
				   					
	__global__
	void proximalF1_RedBlack_Kernel(float* I, const float* I_star, const float* weight_mask, const float* z_I, 
									   const int width, const int height, const int nChannels, const float alpha, const float lambda, const float omega, const bool redKernel);
									   
	__global__
	void proximalF2_Kernel(float* I, const float* z_I, const float* warpI,
									   const int width, const int height, const int nChannels, const float gamma, const float lambda);
									  
									 
	/*******************  CUDA functions *************************************/
						
	void cu_Proximal_F1(float* I, const float* I_star, const float* weight_mask, const float* z_I,
				const int width, const int height, const int nChannels, const float alpha, const float lambda, const int nSORIter);
						
	void cu_Proximal_F2(float* I, const float* I_star, const float* z_I, const float* warpI, 
				const int width, const int height, const int nChannels, const float gamma, const float lambda, const int nSORIter);
						
	void cu_MergeDensity_ADMM_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I, 
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter);
							  
	void cu_MergeDensity_ADMM_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, const float* next_I,
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter);
							  
	void cu_MergeDensity_ADMM_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, const float* next_I,
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter);
							  		
}

#endif