#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_CUH_
#define _ZQ_CUDA_POISSON_SOLVER_2D_CUH_

#include "ZQlibCudaDefines.cuh"


namespace ZQ_CUDA_PoissonSolver2D
{
	/*********************  CUDA functions   *************************/
	
	void cu_Regular_to_MAC_vel(float* mac_u, float* mac_v, const float* u, const float* v, const int width, const int height);
	
	void cu_MAC_to_Regular_vel(float* u, float* v, const float* mac_u, const float* mac_v, const int width, const int height);
	
	/******* OpenPoisson *********/
	void cu_SolveOpenPoissonRedBlack_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlack_Regular(float* u, float* v, const int width, const int height, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
													const int width ,const int height, const int maxIter);
													
	/*Another Implementation of OpenPoisson*/
	void cu_SolveOpenPoissonRedBlack2_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlack2_Regular(float* u, float* v, const int width, const int height, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlackwithOccupy2_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlackwithFaceRatio2_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
													const int width ,const int height, const int maxIter);
													


	/*********************  Kernel functions       *************************/
	__global__
	void Regular_to_MAC_u_Kernel(float* mac_u, const float* u, const int width, const int height);

	__global__
	void Regular_to_MAC_v_Kernel(float* mac_v, const float* v, const int width, const int height);

	__global__
	void MAC_to_Regular_vel_Kernel(float* u, float* v, const float* mac_u, const float* mac_v, const int width, const int height);

	__global__
	void Calculate_Divergence_of_MAC_Kernel(float* divergence, const float* mac_u, const float* mac_v, const int width, const int height);

	__global__
	void Calculate_Divergence_of_MAC_FaceRatio_Kernel(float* divergence, const float* mac_u, const float* mac_v, const float* unoccupyU, const float* unoccupyV, 
											const int width, const int height);


	/*******  OpenPoisson kernels *********/
	__global__
	void Adjust_MAC_u_OpenPoisson_Kernel(float* mac_u, const float* p, const int width, const int height);

	__global__
	void Adjust_MAC_v_OpenPoisson_Kernel(float* mac_v, const float* p, const int width, const int height);

	__global__
	void Adjust_MAC_u_OpenPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height);
	
	__global__
	void Adjust_MAC_v_OpenPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height);
	
	__global__
	void Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height);
	
	__global__
	void Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height);

	
	/*First Implementation*/
	__global__
	void SolvePressure_OpenPoisson_RedBlack_Kernel(float* p, const float* divergence, const int width, const int height, const bool redkernel);

	__global__
	void SolvePressure_OpenPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const int width, const int height, const bool redkernel);

	__global__
	void SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
										const int width, const int height, const bool redkernel);

	/*Another Implementation*/
	__global__
	void SolvePressure_OpenPoisson_RedBlack2_Kernel(float* p, const float* divergence, const int width, const int height, const bool redkernel);

	__global__
	void SolvePressure_OpenPoisson_occupy_RedBlack2_Kernel(float* p, const float* divergence, const bool* occupy, const int width, const int height, const bool redkernel);

	__global__
	void SolvePressure_OpenPoisson_FaceRatio_RedBlack2_Kernel(float* p, const float* divergence, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
										const int width, const int height, const bool redkernel);		
}

#endif