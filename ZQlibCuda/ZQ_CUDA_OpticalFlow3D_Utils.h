#ifndef _ZQ_CUDA_OPTICAL_FLOW_3D_UTILS_H_
#define _ZQ_CUDA_OPTICAL_FLOW_3D_UTILS_H_

namespace ZQ_CUDA_OpticalFlow3D
{
	extern "C"
		void InitDevice3D(const int deviceid);

	extern "C"
	float OpticalFlow3D_L2(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
		const float alpha, const float beta, const int nOuterFPIter, const int nSORIter);

	extern "C"
	float OpticalFlow3D_L1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
		const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter,const int nSORIter);

	extern "C"
	float OpticalFlow3D_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
		const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter);

	extern "C"
	float OpticalFlow3D_ADMM(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
		const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
		const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, 
		const int nSORIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
		const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
		const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_DL1_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
		const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
		const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
		const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
		const float alpha, const float beta, const float gamma, const float lambda, 
		const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_DL1_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
		const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
		const float alpha, const float beta, const float gamma, const float lambda, 
		const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
		const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
		const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);

	extern "C"
	float OpticalFlow3D_ADMM_DL1_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
		const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
		const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
}

#endif