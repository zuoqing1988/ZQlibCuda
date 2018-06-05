#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_CLOSED_POISSON_CUH_
#define _ZQ_CUDA_POISSON_SOLVER_2D_CLOSED_POISSON_CUH_

#include "ZQ_CUDA_PoissonSolver2D.cuh"

namespace ZQ_CUDA_PoissonSolver2D
{
	/*********************  CUDA functions   *************************/
	
	/******* ClosedPoisson *********/
	void cu_SolveClosedPoissonRedBlack_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int maxIter);
	
	void cu_SolveClosedPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, const bool* occupy, /*const int first_x, const int first_y,*/ const float div_per_volume,
										 const int width, const int height, const int maxIter);
	
	void cu_SolveClosedPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, 
													 /*const int first_x, const int first_y,*/ const float div_per_volume, const int width ,const int height, const int maxIter);
													 
	/*********************  Kernel functions       *************************/
	
	/*******  ClosedPoisson kernels *********/
	__global__
	void Adjust_MAC_u_ClosedPoisson_Kernel(float* mac_u, const float* p, const int width, const int height);

	__global__
	void Adjust_MAC_v_ClosedPoisson_Kernel(float* mac_v, const float* p, const int width, const int height);

	__global__
	void Adjust_MAC_u_ClosedPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height);
	
	__global__
	void Adjust_MAC_v_ClosedPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height);
	
	__global__
	void Adjust_MAC_u_ClosedPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height);
	
	__global__
	void Adjust_MAC_v_ClosedPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height);
	
	__global__
	void SolvePressure_ClosedPoisson_RedBlack_Kernel(float* p, const float* divergence, const float div_per_volume, const int width, const int height, const bool redkernel);

	__global__
	void SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, /*const int first_x, const int first_y,*/ const float div_per_volume, 
										const int width, const int height, const bool redkernel);

	__global__
	void SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, 
										/*const int first_x, const int first_y,*/ const float div_per_volume, const int width, const int height, const bool redkernel);

}

#endif