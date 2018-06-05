#ifndef _ZQ_CUDA_MERGE_FLOW_2D_UTILS_CUH_
#define _ZQ_CUDA_MERGE_FLOW_2D_UTILS_CUH_


#include "ZQlibCudaDefines.cuh"
#include <vector_types.h>


namespace ZQ_CUDA_MergeFlow2D
{
	/****************   Base Kernels  **********************************/
	
				   					
	__global__
	void proximalF1_RedBlack_Kernel(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* z_u, const float* z_v,
									   const int width, const int height, const float alpha, const float lambda, const float omega, const bool redKernel);
									  
									 
	/*******************  CUDA functions *************************************/
						
	void cu_Proximal_F1(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* z_u, const float* z_v, 
				const int width, const int height, const float alpha, const float lambda, const int nSORIter);
						
						
	void cu_MergeFlow_ADMM_First(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* next_u, const float* next_v,
							  const int width, const int height, const float alpha, const float gamma, const float lambda, 
							  const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							  
	void cu_MergeFlow_ADMM_Middle(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height,
							   const float alpha, const float gamma, const float lambda, 
							   const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							   
	
	void cu_MergeFlow_ADMM_Last(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
							  const int width, const int height, const float alpha, const float gamma, const float lambda, 
							  const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							  		
}

#endif