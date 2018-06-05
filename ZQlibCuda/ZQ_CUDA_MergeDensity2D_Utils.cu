#ifndef _ZQ_CUDA_MERGE_DENSITY_2D_UTILS_CU_
#define _ZQ_CUDA_MERGE_DENSITY_2D_UTILS_CU_

#include "ZQ_CUDA_MergeDensity2D_Utils.cuh"
#include "ZQ_CUDA_OpticalFlow2D_Utils.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"

namespace ZQ_CUDA_MergeDensity2D
{
	/****************   Base Kernels  **********************************/
	
	__global__
	void proximalF1_RedBlack_Kernel(float* I, const float* I_star, const float* weight_mask, const float* z_I, 
									   const int width, const int height, const int nChannels, const float alpha, const float lambda, const float omega, const bool redKernel)
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

		int offset_single = i*width+j;
		int SLICE = nChannels*width;
		
		for(int c = 0;c < nChannels;c++)
		{
			int offset = offset_single*nChannels;
			float sigma = 0, coeff = 0;

			if(j > 0)
			{
				sigma += I[offset-nChannels];
				coeff += 1;

			}
			if(j < width-1)
			{
				sigma += I[offset+nChannels];
				coeff += 1;
			}
			if(i > 0)
			{
				sigma += I[offset-SLICE];
				coeff += 1;
			}
			if(i < height-1)
			{
				sigma += I[offset+SLICE];
				coeff += 1;
			}
			sigma *= alpha;
			coeff *= alpha;
			sigma += weight_mask[offset_single]*I_star[offset] + 0.5f*lambda*z_I[offset];
			coeff += weight_mask[offset_single] + 0.5f*lambda;
			I[offset] = (1-omega)*I[offset] + omega/coeff*sigma;
		}
	}
	
	__global__
	void proximalF2_Kernel(float* I,const float* z_I, const float* warpI, 
									   const int width, const int height, const int nChannels, const float gamma, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int i = y;
		int j = x;
	
		int offset_single = i*width+j;
		for(int c = 0;c < nChannels;c++)
		{
			int offset = offset_single*nChannels+c;
			I[offset] = (gamma*warpI[offset]+0.5*lambda*z_I[offset])/(gamma+0.5*lambda);
		}
	}
	
	__global__
	void NO_ADMM_RedBlack_Kernel(float* I, const float* I_star, const float* weight_mask, const float* warpI, 
									   const int width, const int height, const int nChannels, const float alpha, const float gamma, const float omega, const bool redKernel)
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

		int offset_single = i*width+j;
		int SLICE = nChannels*width;
		
		for(int c = 0;c < nChannels;c++)
		{
			int offset = offset_single*nChannels;
			float sigma = 0, coeff = 0;

			if(j > 0)
			{
				sigma += I[offset-nChannels];
				coeff += 1;

			}
			if(j < width-1)
			{
				sigma += I[offset+nChannels];
				coeff += 1;
			}
			if(i > 0)
			{
				sigma += I[offset-SLICE];
				coeff += 1;
			}
			if(i < height-1)
			{
				sigma += I[offset+SLICE];
				coeff += 1;
			}
			sigma *= alpha;
			coeff *= alpha;
			sigma += weight_mask[offset_single]*I_star[offset] + gamma*warpI[offset];
			coeff += weight_mask[offset_single] + gamma;
			I[offset] = (1-omega)*I[offset] + omega/coeff*sigma;
		}
	}
	
	/****************************************************************************************************/
	
	
	void cu_Proximal_F1(float* I, const float* I_star, const float* weight_mask, const float* z_I, 
				const int width, const int height, const int nChannels, const float alpha, const float lambda, const int nSORIter)
	{
		
		/* ProximalF(z_I,\lambda) = minimize_{I} \int {w(|I-I_star|^2+\alpha^2 \int {|\nabla I|^2 } + 0.5*\lambda \int {|I-z_I|^2} 
		*
		* The Euler-Lagrange equation is:
		*  w(I-I_star) + 0.5*\lambda(I-z_I) = \alpha^2 \Delta I 
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
			proximalF1_RedBlack_Kernel<<<gridSize,blockSize>>>(I,I_star,weight_mask,z_I,width,height,nChannels,alpha2,lambda,omega,true);
			proximalF1_RedBlack_Kernel<<<gridSize,blockSize>>>(I,I_star,weight_mask,z_I,width,height,nChannels,alpha2,lambda,omega,false);
		}
		/* red - black solver end */
			
	}
	
	void cu_Proximal_F2(float* I, const float* z_I, const float* warpI, 
				const int width, const int height, const int nChannels, const float gamma, const float lambda)
	{
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
	
		proximalF2_Kernel<<<gridSize,blockSize>>>(I,z_I,warpI,width,height,nChannels,gamma,lambda);
	}
	
	
	void cu_NO_ADMM(float* I, const float* I_star, const float* weight_mask, const float* warpI, 
				const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		// set omega
		float omega = 1.0;
		float alpha2 = alpha*alpha;
	


		/* red - black solver begin */
		for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
		{
			NO_ADMM_RedBlack_Kernel<<<gridSize,blockSize>>>(I,I_star,weight_mask,warpI,width,height,nChannels,alpha2,gamma,omega,true);
			NO_ADMM_RedBlack_Kernel<<<gridSize,blockSize>>>(I,I_star,weight_mask,warpI,width,height,nChannels,alpha2,gamma,omega,false);
		}
		/* red - black solver end */
			
	}
	
	void cu_MergeDensity_ADMM_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I, 
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter)
	{
		float* I_for_F1 = I;
		float* I_for_F2 = 0;
		float* I_for_q = 0;
		float* z_I = 0;
		
		float* warpI = 0;

		checkCudaErrors( cudaMalloc((void**)&I_for_F2,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_for_q,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&z_I,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpI,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_for_F2,I_for_F1,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemset(I_for_q,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(z_I,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(warpI,0,sizeof(float)*width*height*nChannels) );


		float new_gamma = gamma;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpI,I,next_I,u,v,width,height,nChannels);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			 
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_I,I_for_F2,1,I_for_q,-1,width,height,nChannels);	
											
			cu_Proximal_F1(I_for_F1,I_star,weight_mask,z_I,width,height,nChannels,alpha,lambda,nSORIter);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_I,I_for_F2,1,I_for_q,1,width,height,nChannels);
			
			cu_Proximal_F2(I_for_F1,z_I,warpI,width,height,nChannels,new_gamma,lambda);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(I_for_q,I_for_F1,1,width,height,nChannels);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(I_for_q,I_for_F2,-1,width,height,nChannels);
			
		}
		
		checkCudaErrors( cudaFree(I_for_F2) );
		checkCudaErrors( cudaFree(I_for_q) );
		checkCudaErrors( cudaFree(z_I) );
		checkCudaErrors( cudaFree(warpI) );

		 I_for_F1 = 0;
		 I_for_F2 = 0;
		 I_for_q = 0;
		 z_I = 0;
		 warpI = 0;
	}
	
	
	void cu_MergeDensity_ADMM_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, 
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter)
	{
		float* I_for_F1 = I;
		float* I_for_F2 = 0;
		float* I_for_q = 0;
		float* z_I = 0;
		
		float* tmp_u = 0;
		float* tmp_v = 0;
		float* warpI = 0;

		checkCudaErrors( cudaMalloc((void**)&I_for_F2,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_for_q,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&z_I,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpI,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_for_F2,I_for_F1,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemset(I_for_q,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(z_I,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpI,0,sizeof(float)*width*height*nChannels) );


		float new_gamma = gamma;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpI,I,pre_I,tmp_u,tmp_v,width,height,nChannels);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			 
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_I,I_for_F2,1,I_for_q,-1,width,height,nChannels);	
											
			cu_Proximal_F1(I_for_F1,I_star,weight_mask,z_I,width,height,nChannels,alpha,lambda,nSORIter);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_I,I_for_F2,1,I_for_q,1,width,height,nChannels);
			
			cu_Proximal_F2(I_for_F1,z_I,warpI,width,height,nChannels,new_gamma,lambda);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(I_for_q,I_for_F1,1,width,height,nChannels);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(I_for_q,I_for_F2,-1,width,height,nChannels);
			
		}
		
		checkCudaErrors( cudaFree(I_for_F2) );
		checkCudaErrors( cudaFree(I_for_q) );
		checkCudaErrors( cudaFree(z_I) );
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		checkCudaErrors( cudaFree(warpI) );

		 I_for_F1 = 0;
		 I_for_F2 = 0;
		 I_for_q = 0;
		 z_I = 0;
		 tmp_u = 0;
		 tmp_v = 0;
		 warpI = 0;
	}
	
	void cu_MergeDensity_ADMM_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, const float* next_I,
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter)
	{
		float* I_for_F1 = I;
		float* I_for_F2 = 0;
		float* I_for_q = 0;
		float* z_I = 0;
		
		float* tmp_u = 0;
		float* tmp_v = 0;
		float* warpI = 0;
		float* tmp_warpI = 0;

		checkCudaErrors( cudaMalloc((void**)&I_for_F2,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_for_q,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&z_I,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpI,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&tmp_warpI,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_for_F2,I_for_F1,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemset(I_for_q,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(z_I,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpI,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(tmp_warpI,0,sizeof(float)*width*height*nChannels) );


		float new_gamma = 2*gamma;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(tmp_warpI,I,next_I,u,v,width,height,nChannels);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(warpI,tmp_warpI,0.5,width,height,nChannels);
		
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(tmp_warpI,I,pre_I,tmp_u,tmp_v,width,height,nChannels);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(warpI,tmp_warpI,0.5,width,height,nChannels);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			 
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_I,I_for_F2,1,I_for_q,-1,width,height,nChannels);	
											
			cu_Proximal_F1(I_for_F1,I_star,weight_mask,z_I,width,height,nChannels,alpha,lambda,nSORIter);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_I,I_for_F2,1,I_for_q,1,width,height,nChannels);
			
			cu_Proximal_F2(I_for_F1,z_I,warpI,width,height,nChannels,new_gamma,lambda);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(I_for_q,I_for_F1,1,width,height,nChannels);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(I_for_q,I_for_F2,-1,width,height,nChannels);
			
		}
		
		checkCudaErrors( cudaFree(I_for_F2) );
		checkCudaErrors( cudaFree(I_for_q) );
		checkCudaErrors( cudaFree(z_I) );
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		checkCudaErrors( cudaFree(warpI) );
		checkCudaErrors( cudaFree(tmp_warpI) );

		 I_for_F1 = 0;
		 I_for_F2 = 0;
		 I_for_q = 0;
		 z_I = 0;
		 tmp_u = 0;
		 tmp_v = 0;
		 warpI = 0;
		 tmp_warpI = 0;
	}
	
	
	void cu_MergeDensity_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I, 
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		float* warpI = 0;
		checkCudaErrors( cudaMalloc((void**)&warpI,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(warpI,0,sizeof(float)*width*height*nChannels) );

		float new_gamma = gamma;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpI,I,next_I,u,v,width,height,nChannels);
		
		
		cu_NO_ADMM(I,I_star,weight_mask,warpI,width,height,nChannels,alpha,new_gamma,nSORIter);
		
		checkCudaErrors( cudaFree(warpI) );
		warpI = 0;
	}
	
	void cu_MergeDensity_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, 
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		float* tmp_u = 0;
		float* tmp_v = 0;
		float* warpI = 0;
		
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpI,sizeof(float)*width*height*nChannels) );
		
		checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpI,0,sizeof(float)*width*height*nChannels) );

		float new_gamma = gamma;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpI,I,pre_I,tmp_u,tmp_v,width,height,nChannels);
		
		cu_NO_ADMM(I,I_star,weight_mask,warpI,width,height,nChannels,alpha,new_gamma,nSORIter);
		
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		checkCudaErrors( cudaFree(warpI) );

		tmp_u = 0;
		tmp_v = 0;
		warpI = 0;
	}
	
	
	
	void cu_MergeDensity_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, const float* next_I,
							  const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		
		float* tmp_u = 0;
		float* tmp_v = 0;
		float* warpI = 0;
		float* tmp_warpI = 0;

		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpI,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&tmp_warpI,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpI,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(tmp_warpI,0,sizeof(float)*width*height*nChannels) );


		float new_gamma = 2*gamma;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(tmp_warpI,I,next_I,u,v,width,height,nChannels);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(warpI,tmp_warpI,0.5,width,height,nChannels);
		
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,1);
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(tmp_warpI,I,pre_I,tmp_u,tmp_v,width,height,nChannels);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(warpI,tmp_warpI,0.5,width,height,nChannels);
		
		cu_NO_ADMM(I,I_star,weight_mask,warpI,width,height,nChannels,alpha,new_gamma,nSORIter);
		
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		checkCudaErrors( cudaFree(warpI) );
		checkCudaErrors( cudaFree(tmp_warpI) );

		 tmp_u = 0;
		 tmp_v = 0;
		 warpI = 0;
		 tmp_warpI = 0;
	}
	
	/***********************************************************************/
	
	

	extern "C"
	float MergeDensity2D_ADMM_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I,
						const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* I_d = 0;
		float* I_star_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* weight_mask_d = 0;
		float* next_I_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_star_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_I_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_d,I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(I_star_d,I_star,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_I_d,next_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_MergeDensity_ADMM_First(I_d,I_star_d,u_d,v_d,weight_mask_d,next_I_d,width,height,nChannels,alpha,gamma,lambda,ADMMIter,nSORIter);

		checkCudaErrors( cudaMemcpy(I,I_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(I_d) );
		checkCudaErrors( cudaFree(I_star_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(next_I_d) );
		
		I_d = 0;
		I_star_d = 0;
		u_d = 0;
		v_d = 0;
		weight_mask_d = 0;
		next_I_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}

	extern "C"
	float MergeDensity2D_ADMM_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I,
						const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* I_d = 0;
		float* I_star_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* weight_mask_d = 0;
		float* pre_I_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_star_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_I_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_d,I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(I_star_d,I_star,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_I_d,pre_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_MergeDensity_ADMM_Last(I_d,I_star_d,u_d,v_d,weight_mask_d,pre_I_d,width,height,nChannels,alpha,gamma,lambda,ADMMIter,nSORIter);

		checkCudaErrors( cudaMemcpy(I,I_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(I_d) );
		checkCudaErrors( cudaFree(I_star_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(pre_I_d) );
		
		I_d = 0;
		I_star_d = 0;
		u_d = 0;
		v_d = 0;
		weight_mask_d = 0;
		pre_I_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	extern "C"
	float MergeDensity2D_ADMM_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, const float* next_I,
						const int width, const int height, const int nChannels, const float alpha, const float gamma, const float lambda, const int ADMMIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* I_d = 0;
		float* I_star_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* weight_mask_d = 0;
		float* pre_I_d = 0;
		float* next_I_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_star_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_I_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_d,I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(I_star_d,I_star,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_I_d,pre_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_I_d,next_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_MergeDensity_ADMM_Middle(I_d,I_star_d,u_d,v_d,weight_mask_d,pre_I_d,next_I_d,width,height,nChannels,alpha,gamma,lambda,ADMMIter,nSORIter);

		checkCudaErrors( cudaMemcpy(I,I_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(I_d) );
		checkCudaErrors( cudaFree(I_star_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(pre_I_d) );
		checkCudaErrors( cudaFree(next_I_d) );
		
		I_d = 0;
		I_star_d = 0;
		u_d = 0;
		v_d = 0;
		weight_mask_d = 0;
		pre_I_d = 0;
		next_I_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	extern "C"
	float MergeDensity2D_First(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* next_I,
						const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* I_d = 0;
		float* I_star_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* weight_mask_d = 0;
		float* next_I_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_star_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_I_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_d,I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(I_star_d,I_star,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_I_d,next_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_MergeDensity_First(I_d,I_star_d,u_d,v_d,weight_mask_d,next_I_d,width,height,nChannels,alpha,gamma,nSORIter);

		checkCudaErrors( cudaMemcpy(I,I_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(I_d) );
		checkCudaErrors( cudaFree(I_star_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(next_I_d) );
		
		I_d = 0;
		I_star_d = 0;
		u_d = 0;
		v_d = 0;
		weight_mask_d = 0;
		next_I_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}

	extern "C"
	float MergeDensity2D_Last(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I,
						const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* I_d = 0;
		float* I_star_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* weight_mask_d = 0;
		float* pre_I_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_star_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_I_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_d,I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(I_star_d,I_star,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_I_d,pre_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_MergeDensity_Last(I_d,I_star_d,u_d,v_d,weight_mask_d,pre_I_d,width,height,nChannels,alpha,gamma,nSORIter);

		checkCudaErrors( cudaMemcpy(I,I_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(I_d) );
		checkCudaErrors( cudaFree(I_star_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(pre_I_d) );
		
		I_d = 0;
		I_star_d = 0;
		u_d = 0;
		v_d = 0;
		weight_mask_d = 0;
		pre_I_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	extern "C"
	float MergeDensity2D_Middle(float* I, const float* I_star, const float* u, const float* v, const float* weight_mask, const float* pre_I, const float* next_I,
						const int width, const int height, const int nChannels, const float alpha, const float gamma, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* I_d = 0;
		float* I_star_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* weight_mask_d = 0;
		float* pre_I_d = 0;
		float* next_I_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&I_star_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&weight_mask_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_I_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_I_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(I_d,I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(I_star_d,I_star,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(weight_mask_d,weight_mask,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_I_d,pre_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_I_d,next_I,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_MergeDensity_Middle(I_d,I_star_d,u_d,v_d,weight_mask_d,pre_I_d,next_I_d,width,height,nChannels,alpha,gamma,nSORIter);

		checkCudaErrors( cudaMemcpy(I,I_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		
		checkCudaErrors( cudaFree(I_d) );
		checkCudaErrors( cudaFree(I_star_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(weight_mask_d) );
		checkCudaErrors( cudaFree(pre_I_d) );
		checkCudaErrors( cudaFree(next_I_d) );
		
		I_d = 0;
		I_star_d = 0;
		u_d = 0;
		v_d = 0;
		weight_mask_d = 0;
		pre_I_d = 0;
		next_I_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
}

#endif