#ifndef _ZQ_CUDA_MERGE_FLOW_2D_UTILS_CU_
#define _ZQ_CUDA_MERGE_FLOW_2D_UTILS_CU_

#include "ZQ_CUDA_MergeFlow2D_Utils.cuh"
#include "ZQ_CUDA_OpticalFlow2D_Utils.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"
#include "ZQ_CUDA_PoissonSolver2D.cuh"

namespace ZQ_CUDA_MergeFlow2D
{
	/****************   Base Kernels  **********************************/
	
	__global__
	void proximalF1_RedBlack_Kernel(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* z_u, const float* z_v,
									   const int width, const int height, const float alpha, const float lambda, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redKernel ? 1 : 0))
			return;


		int i = y;
		int j = x;

		int offset = i * width + j;
		float sigma1 = 0, sigma2 = 0, coeff = 0;

		if(j > 0)
		{
			sigma1 += u[offset-1];
			sigma2 += v[offset-1];
			coeff += 1;

		}
		if(j < width-1)
		{
			sigma1 += u[offset+1];
			sigma2 += v[offset+1];
			coeff  += 1;
		}
		if(i > 0)
		{
			sigma1 += u[offset-width];
			sigma2 += v[offset-width];
			coeff  += 1;
		}
		if(i < height-1)
		{
			sigma1  += u[offset+width];
			sigma2  += v[offset+width];
			coeff   += 1;
		}
		sigma1 *= alpha;
		sigma2 *= alpha;
		coeff *= alpha;
		sigma1 += weight_mask[offset]*u_star[offset] + 0.5f*lambda*z_u[offset];
		float coeff1 = coeff + weight_mask[offset] + 0.5f*lambda;
		u[offset] = (1-omega)*u[offset] + omega/coeff1*sigma1;
		// compute v
		sigma2 += weight_mask[offset]*v_star[offset] + 0.5f*lambda*z_v[offset];
		float coeff2 = coeff + weight_mask[offset] + 0.5f*lambda;
		v[offset] = (1-omega)*v[offset] + omega/coeff2*sigma2;

	}
	
	/****************************************************************************************************/
	
	
	void cu_Proximal_F1(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* z_u, const float* z_v, 
				const int width, const int height, const float alpha, const float lambda, const int nSORIter)
	{
		
		/* ProximalF(z_u,z_v,\lambda) = minimize_{u,v} \int {w(|u-u_star|^2+|v-v_star|^2)} + \alpha^2 \int {|\nabla u|^2 + |\nabla v|^2} + 0.5*\lambda \int {|u-z_u|^2 + |v-z_v|^2} 
		*
		* The Euler-Lagrange equation is:
		*  w(u-u_star) + 0.5*\lambda(u-z_u) = \alpha^2 \Delta u 
		*  w(v-v_star) + 0.5*\lambda(v-z_v) = \alpha^2 \Delta v
		*/

		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		// set omega
		float omega = 1.0;
		float alpha2 = alpha*alpha;
	


		/* red - black solver begin */
		for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
		{
			proximalF1_RedBlack_Kernel<<<gridSize,blockSize>>>(u,v,u_star,v_star,weight_mask,z_u,z_v,width,height,alpha2,lambda,omega,true);
			proximalF1_RedBlack_Kernel<<<gridSize,blockSize>>>(u,v,u_star,v_star,weight_mask,z_u,z_v,width,height,alpha2,lambda,omega,false);
		}
		/* red - black solver end */
			
	}
	
	
	void cu_MergeFlow_ADMM_First(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* next_u, const float* next_v,
							  const int width, const int height, const float alpha, const float gamma, const float lambda, 
							  const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		float new_gamma = gamma*alpha*alpha;

		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1(u_for_F1,v_for_F1,u_star,v_star,weight_mask,z_u,z_v,width,height,alpha,lambda,nSORIter);

			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			ZQ_CUDA_OpticalFlow2D::cu_Proximal_F2_first(u_for_F2,v_for_F2,z_u,z_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);
		
			ZQ_CUDA_OpticalFlow2D::cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			ZQ_CUDA_OpticalFlow2D::cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}
		
		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		 u_for_F1 = 0;
		 v_for_F1 = 0;
		 u_for_F2 = 0;
		 v_for_F2 = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q1 = 0;
		 v_for_q1 = 0;
		 u_for_q2 = 0;
		 v_for_q2 = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	
	void cu_MergeFlow_ADMM_Middle(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, 
							   const float alpha, const float gamma, const float lambda, 
							   const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		

		float new_gamma = gamma*alpha*alpha;


		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);
			
			cu_Proximal_F1(u_for_F1,v_for_F1,u_star,v_star,weight_mask,z_u,z_v,width,height,alpha,lambda,nSORIter);

			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			ZQ_CUDA_OpticalFlow2D::cu_Proximal_F2_middle(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			ZQ_CUDA_OpticalFlow2D::cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			ZQ_CUDA_OpticalFlow2D::cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
	}
	
	
	void cu_MergeFlow_ADMM_Last(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
							  const int width, const int height, const float alpha, const float gamma, const float lambda, 
							  const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		float new_gamma = gamma*alpha*alpha;
		
		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1(u_for_F1,v_for_F1,u_star,v_star,weight_mask,z_u,z_v,width,height,alpha,lambda,nSORIter);

			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			ZQ_CUDA_OpticalFlow2D::cu_Proximal_F2_last(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			ZQ_CUDA_OpticalFlow2D::cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			ZQ_CUDA_OpticalFlow2D::cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			ZQ_CUDA_OpticalFlow2D::cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
			
	}
	
	
	
	/***********************************************************************/
	
	

	extern "C"
	float MergeFlow2D_ADMM_First(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* next_u, const float* next_v,
							  const int width, const int height, const float alpha, const float gamma, const float lambda, 
							  const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_star_d = 0;
		float* v_star_d = 0;
		float* weight_mask_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_star_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_star_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_star_d,u_star,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_star_d,v_star,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_MergeFlow_ADMM_First(u_d,v_d,u_star_d,v_star_d,weight_mask_d,next_u_d,next_v_d,width,height,alpha,gamma,lambda,
									ADMMIter,nSORIter,nWarpFPIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(u_star_d) );
		checkCudaErrors( cudaFree(v_star_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		
		u_d = 0;
		v_d = 0;
		u_star_d = 0;
		v_star_d = 0;
		weight_mask_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}

	
	
	extern "C"
	float MergeFlow2D_ADMM_Middle(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, 
							   const float alpha, const float gamma, const float lambda, 
							   const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_star_d = 0;
		float* v_star_d = 0;
		float* weight_mask_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_star_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_star_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_star_d,u_star,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_star_d,v_star,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_MergeFlow_ADMM_Middle(u_d,v_d,u_star_d,v_star_d,weight_mask_d,pre_u_d,pre_v_d,next_u_d,next_v_d,width,height,
									alpha,gamma,lambda,ADMMIter,nSORIter,nWarpFPIter,nPoissonIter);
	
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(u_star_d) );
		checkCudaErrors( cudaFree(v_star_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		
		u_star_d = 0;
		v_star_d = 0;
		weight_mask_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		u_d = 0;
		v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	
	
	extern "C"
	float MergeFlow2D_ADMM_Last(float* u, float* v, const float* u_star, const float* v_star, const float* weight_mask, const float* pre_u, const float* pre_v,
							  const int width, const int height, const float alpha, const float gamma, const float lambda, 
							  const int ADMMIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_star_d = 0;
		float* v_star_d = 0;
		float* weight_mask_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		
		
		checkCudaErrors( cudaMalloc((void**)&u_star_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_star_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_star_d,u_star,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_star_d,v_star,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_MergeFlow_ADMM_Last(u_d,v_d,u_star_d,v_star_d,weight_mask_d,pre_u_d,pre_v_d,width,height,alpha,gamma,lambda,
							ADMMIter,nSORIter,nWarpFPIter,nPoissonIter);
		
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(u_star_d) );
		checkCudaErrors( cudaFree(v_star_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		
		u_d = 0;
		v_d = 0;
		u_star_d = 0;
		v_star_d = 0;
		weight_mask_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	
}

#endif