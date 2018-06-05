#ifndef _ZQ_CUDA_POISSON_SOLVER_3D_CUH_
#define _ZQ_CUDA_POISSON_SOLVER_3D_CUH_

#include "ZQlibCudaDefines.cuh"

namespace ZQ_CUDA_PoissonSolver3D
{
	/*********************  CUDA functions   *************************/
	void cu_Regular_to_MAC_vel(float* mac_u, float* mac_v, float* mac_w, const float* u, const float* v, const float* w, const int width, const int height, const int depth);
	
	void cu_MAC_to_Regular_vel(float* u, float* v, float* w, const float* mac_u, const float* mac_v, const float* mac_w, const int width, const int height, const int depth);
	
	/*First Implementation*/
	void cu_SolveOpenPoissonRedBlack_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlack_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth, const int maxIter);
	
	void cu_SolveOpenPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
													const int width ,const int height, const int depth, const int maxIter);
													
	/*********************  Kernel functions       *************************/
	__global__
	void Regular_to_MAC_u_Kernel(float* mac_u, const float* u, const int width, const int height, const int depth);
	
	__global__
	void Regular_to_MAC_v_Kernel(float* mac_v, const float* v, const int width, const int height, const int depth);
	
	__global__
	void Regular_to_MAC_w_Kernel(float* mac_w, const float* w, const int width, const int height, const int depth);
	
	__global__
	void MAC_to_Regular_vel_Kernel(float* u, float* v, float* w, const float* mac_u, const float* mac_v, const float* mac_w, const int width, const int height, const int depth);
	
	__global__
	void Calculate_Divergence_of_MAC_Kernel(float* divergence, const float* mac_u, const float* mac_v, const float* mac_w, const int width, const int height, const int depth);
	
	__global__
	void Calculate_Divergence_of_MAC_FaceRatio_Kernel(float* divergence, const float* mac_u, const float* mac_v, const float* mac_w, 
								const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
								const int width, const int height, const int depth);
								
	__global__
	void Adjust_MAC_u_OpenPoisson_Kernel(float* mac_u, const float* p, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_v_OpenPoisson_Kernel(float* mac_v, const float* p, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_w_OpenPoisson_Kernel(float* mac_w, const float* p, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_u_OpenPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_v_OpenPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_w_OpenPoisson_occupy_Kernel(float* mac_w, const float* p, const bool* occupy, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height, const int depth);
	
	__global__
	void Adjust_MAC_w_OpenPoisson_FaceRatio_Kernel(float* mac_w, const float* p, const float* unoccupyW, const int width, const int height, const int depth);
	
	/*First Implementation*/
	__global__
	void SolvePressure_OpenPoisson_RedBlack_Kernel(float* p, const float* divergence, const int width, const int height, const int depth, const bool redkernel);
	
	__global__
	void SolvePressure_OpenPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const int width, const int height, const int depth, const bool redkernel);
	
	__global__
	void SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const int width, const int height, const int depth, const bool redkernel);
										
	
	
}
#endif