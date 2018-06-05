#ifndef _ZQ_CUDA_MERGE_FLOW_2D_UTILS_H_
#define _ZQ_CUDA_MERGE_FLOW_2D_UTILS_H_

namespace ZQ_CUDA_MergeFlow2D
{
	extern "C"
		float MergeFlow2D_ADMM_First(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* next_u, const float* next_v,
		const int width, const int height, const float alpha, const float gama, const float lambda, 
		const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);


	extern "C"
		float MergeFlow2D_ADMM_Middle(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
		const float* next_u, const float* next_v, const int width, const int height, 
		const float alpha, const float gama, const float lambda, 
		const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

	

	extern "C"
		float MergeFlow2D_ADMM_Last(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
		const int width, const int height, const float alpha, const float gamma, const float lambda, 
		const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

}

#endif