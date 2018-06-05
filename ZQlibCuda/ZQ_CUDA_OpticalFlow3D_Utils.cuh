#ifndef _ZQ_CUDA_OPTICAL_FLOW_3D_UTILS_CUH_
#define _ZQ_CUDA_OPTICAL_FLOW_3D_UTILS_CUH_

#include "ZQlibCudaDefines.cuh"

#ifndef optical_flow_L1_eps
#define optical_flow_L1_eps 1e-6
#endif

namespace ZQ_CUDA_OpticalFlow3D
{
	/****************   Base Kernels  **********************************/
	__global__
	void compute_psi_data_Kernel(float* psi_data, const float* imdx, const float* imdy, const float* imdz, const float* imdt, 
							const float* du, const float* dv, const float* dw, const float eps, const int width, const int height, const int depth, const int nChannels);
	
	__global__
	void compute_psi_smooth_Kernel(float* psi_smooth, const float* u, const float* v, const float* w, const float eps, const int width, const int height, const int depth);
	
	__global__
	void compute_psi_u_v_w_Kernel(float* psi_u, float* psi_v, float* psi_w, const float* u, const float* v, const float* w, const float eps, const int width, const int height, const int depth);
	
	__global__
	void compute_imdxdx_imdtdx_Kernel(float* imdxdx, float* imdxdy, float* imdxdz, float* imdydy, float* imdydz, float* imdzdz, 
									float* imdtdx, float* imdtdy, float* imdtdz, const float* imdx, const float* imdy, const float* imdz, const float* imdt,
									const int width, const int height, const int depth, const int nChannels);
									
	__global__
	void compute_imdxdx_imdtdx_withpsidata_Kernel(float* imdxdx, float* imdxdy, float* imdxdz, float* imdydy, float* imdydz, float* imdzdz,
												float* imdtdx, float* imdtdy, float* imdtdz, const float* imdx, const float* imdy, const float* imdz, const float* imdt,
												const float* psi_data, const int width, const int height, const int depth, const int nChannels);
												
	__global__
	void Laplacian_withpsismooth_Kernel(float* output, const float* input,const float* psi_smooth, const int width, const int height, const int depth);
	
	__global__
	void OpticalFlow_L2_RedBlack_Kernel(float* du, float* dv, float* dw, const float* u, const float* v, const float* w,
									const float* imdxdx, const float* imdxdy, const float* imdxdz, const float* imdydy, const float* imdydz, const float* imdzdz,
									const float* imdtdx, const float* imdtdy, const float* imdtdz, const float* laplace_u, const float* laplace_v, const float* laplace_w,
									const int width, const int height, const int depth,	const float alpha, const float beta, const float omega, const bool redKernel);
									
	__global__
	void OpticalFlow_L1_RedBlack_Kernel(float* du, float* dv, float* dw, const float* u, const float* v, const float* w,
								const float* imdxdx, const float* imdxdy, const float* imdxdz, const float* imdydy, const float* imdydz, const float* imdzdz,
								const float* imdtdx, const float* imdtdy, const float* imdtdz, const float* laplace_u, const float* laplace_v, const float* laplace_w,
								const float* psi_smooth, const float* psi_u, const float* psi_v, const float* psi_w,
								const int width, const int height, const int depth, const float alpha, const float beta, const float omega, const bool redKernel);
								
	/******************************* for ADMM  ********************************************/

	__global__
	void proximalF_RedBlack_Kernel(float* du, float* dv, float* dw, const float* imdxdx, const float* imdxdy, const float* imdxdz, const float* imdydy, const float* imdydz, const float* imdzdz, 
					const float* imdtdx, const float* imdtdy, const float* imdtdz, const float* laplace_u, const float* laplace_v, const float*laplace_w, 
					const float* u, const float* z_u, const float* v, const float* z_v,const float* w, const float* z_w,
					const int width, const int height, const int depth, const float alpha, const float beta, const float lambda, const float omega, const bool redKernel);
					
	__global__
	void proximal_F2_Kernel(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* warpU, const float* warpV, const float* warpW,
							const int width, const int height, const int depth, const float gama, const float lambda);
							
	__global__
	void compute_z_u_z_v_z_w_for_proximal_F1_Kernel(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
								const float* u_for_q1, const float* v_for_q1, const float* w_for_q1, const int width, const int height, const int depth, const float lambda);
								
	__global__
	void compute_z_u_z_v_z_w_for_proximal_F2_Kernel(float* z_u, float* z_v, float* z_w, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
								const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda);
								
	__global__
	void compute_z_u_z_v_z_w_for_proximal_G_Kernel(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
									const float* u_for_F2, const float* v_for_F2, const float* w_for_F2, const float* u_for_q1, const float* v_for_q1, const float* w_for_q1,
									const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda);
									
	__global__
	void update_u_v_w_for_q1_q2_Kernel(float* u_for_q1, float* v_for_q1, float* w_for_q1, float* u_for_q2, float* v_for_q2, float* w_for_q2,
									 const float* u_for_F1, const float* v_for_F1, const float* w_for_F1, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
									 const float* u_for_G, const float* v_for_G, const float* w_for_G, const int width, const int height, const int depth, const float lambda);
									 
	/****************************************************************************************************/
	
	void cu_Compute_z_u_z_v_z_w_for_proximal_F1(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
							const float* u_for_q1, const float* v_for_q1, const float* w_for_q1, const int width, const int height, const int depth, const float lambda);
	
	void cu_Compute_z_u_z_v_z_w_for_proximal_F2(float* z_u, float* z_v, float* z_w, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
							const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda);
							
	void cu_Compute_z_u_z_v_z_w_for_proximal_G(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
							const float* u_for_F2, const float* v_for_F2, const float* w_for_F2, const float* u_for_q1, const float* v_for_q1, const float* w_for_q1,
							const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda);
							
	void cu_Update_u_v_w_for_q1_q2(float* u_for_q1, float* v_for_q1, float* w_for_q1, float* u_for_q2, float* v_for_q2, float* w_for_q2,
							const float* u_for_F1, const float* v_for_F1, const float* w_for_F1, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
							const float* u_for_G, const float* v_for_G, const float* w_for_G, const int width, const int height, const int depth, const float lambda);
							
	void cu_GetDerivatives(float* imdx, float* imdy, float* imdz, float* imdt, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels);
	
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L2(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter);
						
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter,const int nSORIter);
						
	void cu_OpticalFlow_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter);
						
	void cu_Proximal_F1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* z_u, const float* z_v, const float* z_w,
				const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float lambda, const int nOuterFPIter, const int nSORIter);
				
	void cu_Proximal_F1_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* z_u, const float* z_v, const float* z_w,
					const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float lambda, 
					const int nOuterFPIter, const int nInnerFPIter, const int nSORIter);
					
	void cu_Proximal_F2_first(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* next_u, const float* next_v, const float* next_w,
				const int width, const int height, const int depth, const float gama, const float lambda, const int nFPIter, const int nPoissonIter);
				
	void cu_Proximal_F2_middle(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* pre_u, const float* pre_v, const float* pre_w,
						const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, 
						const float gama, const float lambda, const int nFPIter, const int nPoissonIter);
						
	void cu_Proximal_F2_last(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* pre_u, const float* pre_v, const float* pre_w,
							const int width, const int height, const int depth, const float gama, const float lambda, const int nFPIter, const int nPoissonIter);
							
	void cu_Proximal_G(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const int width, const int height, const int depth, const int nPoissonIter);
	
	void cu_OpticalFlow_ADMM(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter);
						
	void cu_OpticalFlow_ADMM_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, 
						const int nSORIter, const int nPoissonIter);
	
	void cu_OpticalFlow_ADMM_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							  
	void cu_OpticalFlow_ADMM_DL1_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							  
	void cu_OpticalFlow_ADMM_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							   const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							   
	void cu_OpticalFlow_ADMM_DL1_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							   const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							   
	void cu_OpticalFlow_ADMM_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							  
	void cu_OpticalFlow_ADMM_DL1_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter);
							  
	
	
}

#endif