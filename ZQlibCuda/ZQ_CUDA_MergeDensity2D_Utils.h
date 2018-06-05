#ifndef _ZQ_CUDA_MERGE_DENSITY_2D_UTILS_H_
#define _ZQ_CUDA_MERGE_DENSITY_2D_UTILS_H_

namespace ZQ_CUDA_MergeDensity2D
{
	
	extern "C"
		float MergeDensity2D_ADMM_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I,
		const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter);


	extern "C"
		float MergeDensity2D_ADMM_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I,
		const float* next_I, const int width, const int height, const int nChannels,const float alpha, const float gamma, const float lambda, 
		const int ADMMIter, const int nSORIter);



	extern "C"
		float MergeDensity2D_ADMM_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I,
		const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter);

	/***DONT USE ADMM***/

	extern "C"
		float MergeDensity2D_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I,
		const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter);


	extern "C"
		float MergeDensity2D_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I,
		const float* next_I, const int width, const int height, const int nChannels,const float alpha, const float gamma, const int nSORIter);



	extern "C"
		float MergeDensity2D_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I,
		const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter);

}

#endif