#ifndef _ZQ_CUDA_POISSON_SOLVER_3D_OPEN_FLUX_CU_
#define _ZQ_CUDA_POISSON_SOLVER_3D_OPEN_FLUX_CU_

#include "ZQ_CUDA_PoissonSolver3D_OpenFlux.cuh"
#include "ZQ_CUDA_ImageProcessing3D.cuh"

namespace ZQ_CUDA_PoissonSolver3D
{
	__global__
	void SolveFlux_OpenFlux_u_RedBlack_Kernel(float* out_du, const float* du, const float* dv, const float* dw, const float* divergence, const float* lambda, const float aug_coeff,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		int rest = x%2;

		if(rest == (redkernel ? 1 : 0))
			return;
			
		for(int z = 0;z < depth;z++)
		{
			float coeff = 2.0f,sigma = 0.0f;
			
			if(x < width)
			{
				sigma -= lambda[z*height*width+y*width+x];
				coeff += aug_coeff;
				sigma += aug_coeff*(du[z*height*(width+1)+y*(width+1)+x+1]+dv[z*(height+1)*width+(y+1)*width+x]-dv[z*(height+1)*width+y*width+x]
									+dw[(z+1)*height*width+y*width+x]-dw[z*height*width+y*width+x]+divergence[z*height*width+y*width+x]);
			}
			
			if(x > 0)
			{
				sigma += lambda[z*height*width+y*width+x-1];
				coeff += aug_coeff;
				sigma -= aug_coeff*(-du[z*height*(width+1)+y*(width+1)+x-1]+dv[z*(height+1)*width+(y+1)*width+x-1]-dv[z*(height+1)*width+y*width+x-1]
									+dw[(z+1)*height*width+y*width+x-1]-dw[z*height*width+y*width+x-1]+divergence[z*height*width+y*width+x-1]);
			}
			out_du[z*height*(width+1)+y*(width+1)+x] = sigma/coeff;
		}
	}
	
	
	__global__
	void SolveFlux_OpenFlux_v_RedBlack_Kernel(float* out_dv, const float* du, const float* dv, const float* dw, const float* divergence, const float* lambda, const float aug_coeff,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		int rest = y%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		for(int z = 0;z < depth;z++)
		{
			float coeff = 2.0f,sigma = 0.0f;
			
			if(y < height)
			{
				sigma -= lambda[z*height*width+y*width+x];
				coeff += aug_coeff;
				sigma += aug_coeff*(du[z*height*(width+1)+y*(width+1)+x+1]-du[z*height*(width+1)+y*(width+1)+x]+dv[z*(height+1)*width+(y+1)*width+x]
									+dw[(z+1)*height*width+y*width+x]-dw[z*height*width+y*width+x]+divergence[z*height*width+y*width+x]);
			}
			
			if(y > 0)
			{
				sigma += lambda[z*height*width+(y-1)*width+x];
				coeff += aug_coeff;
				sigma -= aug_coeff*(du[z*height*(width+1)+(y-1)*(width+1)+x+1]-du[z*height*(width+1)+(y-1)*(width+1)+x]-dv[z*(height+1)*width+(y-1)*width+x]
									+dw[(z+1)*height*width+(y-1)*width+x]-dw[z*height*width+(y-1)*width+x]+divergence[z*height*width+(y-1)*width+x]);
			}
			out_dv[z*(height+1)*width+y*width+x] = sigma/coeff;
		}
	}
	
	__global__
	void SolveFlux_OpenFlux_w_RedBlack_Kernel(float* out_dw, const float* du, const float* dv, const float* dw, const float* divergence, const float* lambda, const float aug_coeff,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int start = redkernel ? 0 : 1;

		for(int z = start;z <= depth;z += 2)
		{
			float coeff = 2.0f,sigma = 0.0f;
			
			if(z < depth)
			{
				sigma -= lambda[z*height*width+y*width+x];
				coeff += aug_coeff;
				sigma += aug_coeff*(du[z*height*(width+1)+y*(width+1)+x+1]-du[z*height*(width+1)+y*(width+1)+x]
									+dv[z*(height+1)*width+(y+1)*width+x]-dv[z*(height+1)*width+y*width+x]
									+dw[(z+1)*height*width+y*width+x]+divergence[z*height*width+y*width+x]);
			}
			
			if(z > 0)
			{
				sigma += lambda[(z-1)*height*width+y*width+x];
				coeff += aug_coeff;
				sigma -= aug_coeff*(du[(z-1)*height*(width+1)+y*(width+1)+x+1]-du[(z-1)*height*(width+1)+y*(width+1)+x]
									+dv[(z-1)*(height+1)*width+(y+1)*width+x]-dv[(z-1)*(height+1)*width+y*width+x]
									-dw[(z-1)*height*width+y*width+x]+divergence[(z-1)*height*width+y*width+x]);
			}
			out_dw[z*height*width+y*width+x] = sigma/coeff;
		}
	}
	
	
	
	__global__
	void SolveFlux_OpenFlux_occupy_u_RedBlack_Kernel(float* out_du, const float* du, const float* dv, const float* dw, const bool* occupy, const float* divergence, const float* lambda, const float aug_coeff,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		int rest = x%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		for(int z = 0;z < depth;z++)
		{
			float coeff = 2.0f,sigma = 0.0f;
			
			if(x < width)
			{
				if(occupy[z*height*width+y*width+x])
					continue ;
				sigma -= lambda[z*height*width+y*width+x];
				coeff += aug_coeff;
				sigma += aug_coeff*(du[z*height*(width+1)+y*(width+1)+x+1]+dv[z*(height+1)*width+(y+1)*width+x]-dv[z*(height+1)*width+y*width+x]
									+dw[(z+1)*height*width+y*width+x]-dw[z*height*width+y*width+x]+divergence[z*height*width+y*width+x]);
			}
			
			if(x > 0)
			{
				if(occupy[z*height*width+y*width+x-1])
					continue ;
				sigma += lambda[z*height*width+y*width+x-1];
				coeff += aug_coeff;
				sigma -= aug_coeff*(-du[z*height*(width+1)+y*(width+1)+x-1]+dv[z*(height+1)*width+(y+1)*width+x-1]-dv[z*(height+1)*width+y*width+x-1]
									+dw[(z+1)*height*width+y*width+x-1]-dw[z*height*width+y*width+x-1]+divergence[z*height*width+y*width+x-1]);
			}
			out_du[z*height*(width+1)+y*(width+1)+x] = sigma/coeff;
		}
	}
	
	
	__global__
	void SolveFlux_OpenFlux_occupy_v_RedBlack_Kernel(float* out_dv, const float* du, const float* dv, const float* dw, const bool* occupy, const float* divergence, const float* lambda, const float aug_coeff,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		int rest = y%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		for(int z = 0;z < depth;z++)
		{
			float coeff = 2.0f,sigma = 0.0f;
			
			if(y < height)
			{
				if(occupy[z*height*width+y*width+x])
					continue ;
				sigma -= lambda[z*height*width+y*width+x];
				coeff += aug_coeff;
				sigma += aug_coeff*(du[z*height*(width+1)+y*(width+1)+x+1]-du[z*height*(width+1)+y*(width+1)+x]+dv[z*(height+1)*width+(y+1)*width+x]
									+dw[(z+1)*height*width+y*width+x]-dw[z*height*width+y*width+x]+divergence[z*height*width+y*width+x]);
			}
			
			if(y > 0)
			{
				if(occupy[z*height*width+(y-1)*width+x])
					continue ;
				sigma += lambda[z*height*width+(y-1)*width+x];
				coeff += aug_coeff;
				sigma -= aug_coeff*(du[z*height*(width+1)+(y-1)*(width+1)+x+1]-du[z*height*(width+1)+(y-1)*(width+1)+x]-dv[z*(height+1)*width+(y-1)*width+x]
									+dw[(z+1)*height*width+(y-1)*width+x]-dw[z*height*width+(y-1)*width+x]+divergence[z*height*width+(y-1)*width+x]);
			}
			out_dv[z*(height+1)*width+y*width+x] = sigma/coeff;
		}
	}
	
	__global__
	void SolveFlux_OpenFlux_occupy_w_RedBlack_Kernel(float* out_dw, const float* du, const float* dv, const float* dw, const bool* occupy, const float* divergence, const float* lambda, const float aug_coeff,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int start = redkernel ? 0 : 1;
		for(int z = start;z <= depth;z += 2)
		{
			float coeff = 2.0f,sigma = 0.0f;
			
			if(z < depth)
			{
				if(occupy[z*height*width+y*width+x])
					continue ;
				sigma -= lambda[z*height*width+y*width+x];
				coeff += aug_coeff;
				sigma += aug_coeff*(du[z*height*(width+1)+y*(width+1)+x+1]-du[z*height*(width+1)+y*(width+1)+x]
									+dv[z*(height+1)*width+(y+1)*width+x]-dv[z*(height+1)*width+y*width+x]
									+dw[(z+1)*height*width+y*width+x]+divergence[z*height*width+y*width+x]);
			}
			
			if(z > 0)
			{
				if(occupy[(z-1)*height*width+y*width+x])
					continue ;
				sigma += lambda[(z-1)*height*width+y*width+x];
				coeff += aug_coeff;
				sigma -= aug_coeff*(du[(z-1)*height*(width+1)+y*(width+1)+x+1]-du[(z-1)*height*(width+1)+y*(width+1)+x]
									+dv[(z-1)*(height+1)*width+(y+1)*width+x]-dv[(z-1)*(height+1)*width+y*width+x]
									-dw[(z-1)*height*width+y*width+x]+divergence[(z-1)*height*width+y*width+x]);
			}
			out_dw[z*height*width+y*width+x] = sigma/coeff;
		}
	}
	
	
	__global__
	void SolveFlux_OpenFlux_FaceRatio_u_RedBlack_Kernel(float* out_du, const float* du, const float* dv, const float* dw, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const float* divergence, const float* lambda, const float aug_coeff, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		int rest = x%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		for(int z = 0;z < depth;z++)
		{
			float ratio = unoccupyU[z*height*(width+1)+y*(width+1)+x];
			float ratio2 = ratio*ratio;
			
			if(ratio == 0)
				continue ;
			
			float coeff = 2.0f*ratio,sigma = 0.0f;
			
			if(x < width)
			{
				sigma -= ratio*lambda[z*height*width+y*width+x];
				coeff += ratio2*aug_coeff;
				sigma += ratio*aug_coeff*(
							unoccupyU[z*height*(width+1)+y*(width+1)+x+1]*du[z*height*(width+1)+y*(width+1)+x+1]
						   +unoccupyV[z*(height+1)*width+(y+1)*width+x]*dv[z*(height+1)*width+(y+1)*width+x]
						   -unoccupyV[z*(height+1)*width+y*width+x]*dv[z*(height+1)*width+y*width+x]
						   +unoccupyW[(z+1)*height*width+y*width+x]*dw[(z+1)*height*width+y*width+x]
						   -unoccupyW[z*height*width+y*width+x]*dw[z*height*width+y*width+x]
						   +divergence[z*height*width+y*width+x]);
			}
			
			if(x > 0)
			{
				sigma += ratio*lambda[z*height*width+y*width+x-1];
				coeff += ratio2*aug_coeff;
				sigma -= ratio*aug_coeff*(
							-unoccupyU[z*height*(width+1)+y*(width+1)+x-1]*du[z*height*(width+1)+y*(width+1)+x-1]
							+unoccupyV[z*(height+1)*width+(y+1)*width+x-1]*dv[z*(height+1)*width+(y+1)*width+x-1]
							-unoccupyV[z*(height+1)*width+y*width+x-1]*dv[z*(height+1)*width+y*width+x-1]
							+unoccupyW[(z+1)*height*width+y*width+x-1]*dw[(z+1)*height*width+y*width+x-1]
							-unoccupyW[z*height*width+y*width+x-1]*dw[z*height*width+y*width+x-1]
							+divergence[z*height*width+y*width+x-1]);
			}
			out_du[z*height*(width+1)+y*(width+1)+x] = sigma/coeff;
		}
	}
	
	__global__
	void SolveFlux_OpenFlux_FaceRatio_v_RedBlack_Kernel(float* out_dv, const float* du, const float* dv, const float* dw, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const float* divergence, const float* lambda, const float aug_coeff, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		int rest = y%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		for(int z = 0;z < depth;z++)
		{
			float ratio = unoccupyV[z*(height+1)*width+y*width+x];
			
			if(ratio == 0)
				continue ;
				
			float ratio2 = ratio*ratio;
			float coeff = 2.0f*ratio,sigma = 0.0f;
			
			if(y < height)
			{
				sigma -= ratio*lambda[z*height*width+y*width+x];
				coeff += ratio2*aug_coeff;
				sigma += ratio*aug_coeff*(
							unoccupyU[z*height*(width+1)+y*(width+1)+x+1]*du[z*height*(width+1)+y*(width+1)+x+1]
						   -unoccupyU[z*height*(width+1)+y*(width+1)+x]*du[z*height*(width+1)+y*(width+1)+x]
						   +unoccupyV[z*(height+1)*width+(y+1)*width+x]*dv[z*(height+1)*width+(y+1)*width+x]
						   +unoccupyW[(z+1)*height*width+y*width+x]*dw[(z+1)*height*width+y*width+x]
						   -unoccupyW[z*height*width+y*width+x]*dw[z*height*width+y*width+x]
						   +divergence[z*height*width+y*width+x]);
			}
			
			if(y > 0)
			{
				sigma += ratio*lambda[z*height*width+(y-1)*width+x];
				coeff += ratio2*aug_coeff;
				sigma -= ratio*aug_coeff*(
							unoccupyU[z*height*(width+1)+(y-1)*(width+1)+x+1]*du[z*height*(width+1)+(y-1)*(width+1)+x+1]
						   -unoccupyU[z*height*(width+1)+(y-1)*(width+1)+x]*du[z*height*(width+1)+(y-1)*(width+1)+x]
						   -unoccupyV[z*(height+1)*width+(y-1)*width+x]*dv[z*(height+1)*width+(y-1)*width+x]
						   +unoccupyW[(z+1)*height*width+(y-1)*width+x]*dw[(z+1)*height*width+(y-1)*width+x]
						   -unoccupyW[z*height*width+(y-1)*width+x]*dw[z*height*width+(y-1)*width+x]
						   +divergence[z*height*width+(y-1)*width+x]);
			}
			out_dv[z*(height+1)*width+y*width+x] = sigma/coeff;
		}
	}
	
	__global__
	void SolveFlux_OpenFlux_FaceRatio_w_RedBlack_Kernel(float* out_dw, const float* du, const float* dv, const float* dw, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const float* divergence, const float* lambda, const float aug_coeff, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int start = redkernel ? 0 : 1;

		for(int z = start;z <= depth;z += 2)
		{
			float ratio = unoccupyW[z*height*width+y*width+x];
			
			if(ratio == 0)
				continue ;
				
			float ratio2 = ratio*ratio;
			float coeff = 2.0f*ratio,sigma = 0.0f;
			
			if(z < depth)
			{
				sigma -= ratio*lambda[z*height*width+y*width+x];
				coeff += ratio2*aug_coeff;
				sigma += ratio*aug_coeff*(
							unoccupyU[z*height*(width+1)+y*(width+1)+x+1]*du[z*height*(width+1)+y*(width+1)+x+1]
						   -unoccupyU[z*height*(width+1)+y*(width+1)+x]*du[z*height*(width+1)+y*(width+1)+x]
						   +unoccupyV[z*(height+1)*width+(y+1)*width+x]*dv[z*(height+1)*width+(y+1)*width+x]
						   -unoccupyV[z*(height+1)*width+y*width+x]*dv[z*(height+1)*width+y*width+x]
						   +unoccupyW[(z+1)*height*width+y*width+x]*dw[(z+1)*height*width+y*width+x]
						   +divergence[z*height*width+y*width+x]);
			}
			
			if(z > 0)
			{
				sigma += ratio*lambda[(z-1)*height*width+y*width+x];
				coeff += ratio2*aug_coeff;
				sigma -= ratio*aug_coeff*(
							unoccupyU[(z-1)*height*(width+1)+y*(width+1)+x+1]*du[(z-1)*height*(width+1)+y*(width+1)+x+1]
						   -unoccupyU[(z-1)*height*(width+1)+y*(width+1)+x]*du[(z-1)*height*(width+1)+y*(width+1)+x]
						   +unoccupyV[(z-1)*(height+1)*width+(y+1)*width+x]*dv[(z-1)*(height+1)*width+(y+1)*width+x]
						   -unoccupyV[(z-1)*(height+1)*width+y*width+x]*dv[(z-1)*(height+1)*width+y*width+x]						   
						   -unoccupyW[(z-1)*height*width+y*width+x]*dw[(z-1)*height*width+y*width+x]
						   +divergence[(z-1)*height*width+y*width+x]);
			}
			out_dw[z*height*width+y*width+x] = sigma/coeff;
		}
	}
										
	/********************************************************/
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveOpenFluxRedBlack_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		float* b_d = 0;
		float* tmp_div_d = 0;
		float* lambda_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&lambda_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&tmp_div_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(lambda_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(tmp_div_d,0,sizeof(float)*width*height*depth));
		
		float* du_d = 0;
		float* dv_d = 0;
		float* dw_d = 0;
		float* tmp_du_d = 0;
		float* tmp_dv_d = 0;
		float* tmp_dw_d = 0;
		checkCudaErrors( cudaMalloc((void**)&du_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&tmp_du_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dv_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dw_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(du_d,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(dv_d,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(dw_d,0,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(tmp_du_d,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(tmp_dv_d,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(tmp_dw_d,0,sizeof(float)*width*height*(depth+1)) );
		

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,mac_w,width,height,depth);
		
		const float max_aug_coeff = 1e6;
		float aug_coeff = 1.0f;
		for(int out_it = 0; out_it < outerIter; out_it++)
		{
			//Red-Black Solve du,dv,dw
			for(int rd_it = 0; rd_it < innerIter; rd_it++)
			{
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,dw_d,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,dw_d,b_d,lambda_d,aug_coeff,width,height,depth,false);			
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,dw_d,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,dw_d,b_d,lambda_d,aug_coeff,width,height,depth,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dw_d,dw_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_w_RedBlack_Kernel<<<w_gridSize,blockSize>>>(dw_d,du_d,dv_d,tmp_dw_d,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dw_d,dw_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_w_RedBlack_Kernel<<<w_gridSize,blockSize>>>(dw_d,du_d,dv_d,tmp_dw_d,b_d,lambda_d,aug_coeff,width,height,depth,false);
			}
			
			Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(tmp_div_d,du_d,dv_d,dw_d,width,height,depth);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,b_d,1.0f,width,height,depth,1);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(lambda_d,tmp_div_d,-aug_coeff,width,height,depth,1);
			
			aug_coeff *= 2.0f;
			if(aug_coeff > max_aug_coeff)
				aug_coeff = max_aug_coeff;
		}
		
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<u_gridSize,blockSize>>>(mac_u,du_d,1.0f,width+1,height,depth,1);
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<v_gridSize,blockSize>>>(mac_v,dv_d,1.0f,width,height+1,depth,1);
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<w_gridSize,blockSize>>>(mac_w,dw_d,1.0f,width,height,depth+1,1);
		
		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(tmp_div_d) );
		checkCudaErrors( cudaFree(lambda_d) );
		checkCudaErrors( cudaFree(du_d) );
		checkCudaErrors( cudaFree(dv_d) );
		checkCudaErrors( cudaFree(dw_d) );
		checkCudaErrors( cudaFree(tmp_du_d) );
		checkCudaErrors( cudaFree(tmp_dv_d) );
		checkCudaErrors( cudaFree(tmp_dw_d) );
		b_d = 0;
		tmp_div_d = 0;
		lambda_d = 0;
		du_d = 0;
		dv_d = 0;
		dw_d = 0;
		tmp_du_d = 0;
		tmp_dv_d = 0;
		tmp_dw_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveOpenFluxRedBlack_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		float* mac_u = 0;
		float* mac_v = 0;
		float* mac_w = 0;
		checkCudaErrors( cudaMalloc((void**)&mac_u,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(mac_u,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(mac_v,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(mac_w,0,sizeof(float)*width*height*(depth+1)) );

		cu_Regular_to_MAC_vel(mac_u,mac_v,mac_w,u,v,w,width,height,depth);
		cu_SolveOpenFluxRedBlack_MAC(mac_u,mac_v,mac_w,width,height,depth,outerIter,innerIter);
		cu_MAC_to_Regular_vel(u,v,w,mac_u,mac_v,mac_w,width,height,depth);

		checkCudaErrors( cudaFree(mac_u) );
		checkCudaErrors( cudaFree(mac_v) );
		checkCudaErrors( cudaFree(mac_w) );
		mac_u = 0;
		mac_v = 0;	
		mac_w = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveOpenFluxRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		float* b_d = 0;
		float* tmp_div_d = 0;
		float* lambda_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&lambda_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&tmp_div_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(lambda_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(tmp_div_d,0,sizeof(float)*width*height*depth));
		
		float* du_d = 0;
		float* dv_d = 0;
		float* dw_d = 0;
		float* tmp_du_d = 0;
		float* tmp_dv_d = 0;
		float* tmp_dw_d = 0;
		checkCudaErrors( cudaMalloc((void**)&du_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&tmp_du_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dv_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dw_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(du_d,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(dv_d,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(dw_d,0,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(tmp_du_d,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(tmp_dv_d,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(tmp_dw_d,0,sizeof(float)*width*height*(depth+1)) );
		

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,mac_w,width,height,depth);
		
		float aug_coeff = 1.0f;
		float max_aug_coeff = 1e6;
		for(int out_it = 0; out_it < outerIter; out_it++)
		{
			//Red-Black Solve du,dv
			for(int rd_it = 0; rd_it < innerIter; rd_it++)
			{
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_occupy_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,dw_d,occupy,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_occupy_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,dw_d,occupy,b_d,lambda_d,aug_coeff,width,height,depth,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_occupy_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,dw_d,occupy,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_occupy_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,dw_d,occupy,b_d,lambda_d,aug_coeff,width,height,depth,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dw_d,dw_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_occupy_w_RedBlack_Kernel<<<w_gridSize,blockSize>>>(dw_d,du_d,dv_d,tmp_dw_d,occupy,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dw_d,dw_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_occupy_w_RedBlack_Kernel<<<w_gridSize,blockSize>>>(dw_d,du_d,dv_d,tmp_dw_d,occupy,b_d,lambda_d,aug_coeff,width,height,depth,false);
			}
			
			Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(tmp_div_d,du_d,dv_d,dw_d,width,height,depth);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,b_d,1.0f,width,height,depth,1);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(lambda_d,tmp_div_d,-aug_coeff,width,height,depth,1);
			
			aug_coeff *= 2.0f;
			if(aug_coeff > max_aug_coeff)
				aug_coeff = max_aug_coeff;
		}
		
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<u_gridSize,blockSize>>>(mac_u,du_d,1.0f,width+1,height,depth,1);
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<v_gridSize,blockSize>>>(mac_v,dv_d,1.0f,width,height+1,depth,1);
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<w_gridSize,blockSize>>>(mac_w,dw_d,1.0f,width,height,depth+1,1);
		
		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(tmp_div_d) );
		checkCudaErrors( cudaFree(lambda_d) );
		checkCudaErrors( cudaFree(du_d) );
		checkCudaErrors( cudaFree(dv_d) );
		checkCudaErrors( cudaFree(dw_d) );
		checkCudaErrors( cudaFree(tmp_du_d) );
		checkCudaErrors( cudaFree(tmp_dv_d) );
		checkCudaErrors( cudaFree(tmp_dw_d) );
		b_d = 0;
		tmp_div_d = 0;
		lambda_d = 0;
		du_d = 0;
		dv_d = 0;
		dw_d = 0;
		tmp_du_d = 0;
		tmp_dv_d = 0;
		tmp_dw_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveOpenFluxRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* tmp_div_d = 0;
		float* lambda_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&lambda_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&tmp_div_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(lambda_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(tmp_div_d,0,sizeof(float)*width*height*depth));
		
		float* du_d = 0;
		float* dv_d = 0;
		float* dw_d = 0;
		float* tmp_du_d = 0;
		float* tmp_dv_d = 0;
		float* tmp_dw_d = 0;
		checkCudaErrors( cudaMalloc((void**)&du_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&tmp_du_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dv_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dw_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(du_d,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(dv_d,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(dw_d,0,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemset(tmp_du_d,0,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMemset(tmp_dv_d,0,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMemset(tmp_dw_d,0,sizeof(float)*width*height*(depth+1)) );
		

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,mac_w,width,height,depth);
		
		float aug_coeff = 1.0f;
		float max_aug_coeff = 1e6;
		for(int out_it = 0; out_it < outerIter; out_it++)
		{
			//Red-Black Solve du,dv
			for(int rd_it = 0; rd_it < innerIter; rd_it++)
			{
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_FaceRatio_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,dw_d,unoccupyU,unoccupyV,unoccupyW,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_FaceRatio_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,dw_d,unoccupyU,unoccupyV,unoccupyW,b_d,lambda_d,aug_coeff,width,height,depth,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_FaceRatio_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,dw_d,unoccupyU,unoccupyV,unoccupyW,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_FaceRatio_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,dw_d,unoccupyU,unoccupyV,unoccupyW,b_d,lambda_d,aug_coeff,width,height,depth,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dw_d,dw_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_FaceRatio_w_RedBlack_Kernel<<<w_gridSize,blockSize>>>(dw_d,du_d,dv_d,tmp_dw_d,unoccupyU,unoccupyV,unoccupyW,b_d,lambda_d,aug_coeff,width,height,depth,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dw_d,dw_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_OpenFlux_FaceRatio_w_RedBlack_Kernel<<<w_gridSize,blockSize>>>(dw_d,du_d,dv_d,tmp_dw_d,unoccupyU,unoccupyV,unoccupyW,b_d,lambda_d,aug_coeff,width,height,depth,false);
			}
			
			Calculate_Divergence_of_MAC_FaceRatio_Kernel<<<gridSize,blockSize>>>(tmp_div_d,du_d,dv_d,dw_d,unoccupyU,unoccupyV,unoccupyW,width,height,depth);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,b_d,1.0f,width,height,depth,1);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(lambda_d,tmp_div_d,-aug_coeff,width,height,depth,1);
			
			aug_coeff *= 2.0f;
			if(aug_coeff > max_aug_coeff)
				aug_coeff = max_aug_coeff;
		}
		
		
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<u_gridSize,blockSize>>>(mac_u,du_d,1.0f,width+1,height,depth,1);
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<v_gridSize,blockSize>>>(mac_v,dv_d,1.0f,width,height+1,depth,1);
		ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<w_gridSize,blockSize>>>(mac_w,dw_d,1.0f,width,height,depth+1,1);
		
		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(tmp_div_d) );
		checkCudaErrors( cudaFree(lambda_d) );
		checkCudaErrors( cudaFree(du_d) );
		checkCudaErrors( cudaFree(dv_d) );
		checkCudaErrors( cudaFree(dw_d) );
		checkCudaErrors( cudaFree(tmp_du_d) );
		checkCudaErrors( cudaFree(tmp_dv_d) );
		checkCudaErrors( cudaFree(tmp_dw_d) );
		b_d = 0;
		tmp_div_d = 0;
		lambda_d = 0;
		du_d = 0;
		dv_d = 0;
		dw_d = 0;
		tmp_du_d = 0;
		tmp_dv_d = 0;
		tmp_dw_d = 0;
	}
	
	
	/*************************************************************/
	
	/*First Implementation*/
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" 
	void SolveOpenFluxRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );

		cu_SolveOpenFluxRedBlack_MAC(mac_u_d,mac_v_d,mac_w_d,width,height,depth,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_w,mac_w_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(mac_w_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		mac_w_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C"
	void SolveOpenFluxRedBlack3D_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );

		cu_SolveOpenFluxRedBlack_Regular(u_d,v_d,w_d,width,height,depth,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		u_d = 0;
		v_d = 0;
		w_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" 
	void SolveOpenFluxRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth,
									const int outerIter, const int innerIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;
		bool* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height*depth,cudaMemcpyHostToDevice) );

		cu_SolveOpenFluxRedBlackwithOccupy_MAC(mac_u_d,mac_v_d,mac_w_d,occupy_d,width,height,depth,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_w,mac_w_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(mac_w_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		mac_w_d = 0;
		occupy_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" 
	void SolveOpenFluxRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
											const int width, const int height, const int depth, const int outerIter, const int innerIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;
		float* unoccupyU_d = 0;
		float* unoccupyV_d = 0;
		float* unoccupyW_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyU_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyV_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyW_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyU_d,unoccupyU,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyV_d,unoccupyV,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyW_d,unoccupyW,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		
		cu_SolveOpenFluxRedBlackwithFaceRatio_MAC(mac_u_d,mac_v_d,mac_w_d,unoccupyU_d,unoccupyV_d,unoccupyW_d,width,height,depth,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_w,mac_w_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(mac_w_d) );
		checkCudaErrors( cudaFree(unoccupyU_d) );
		checkCudaErrors( cudaFree(unoccupyV_d) );
		checkCudaErrors( cudaFree(unoccupyW_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		mac_w_d = 0;
		unoccupyU_d = 0;
		unoccupyV_d = 0;
		unoccupyW_d = 0;
	}
	
	
}

#endif