#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_CLOSED_FLUX_CU_
#define _ZQ_CUDA_POISSON_SOLVER_2D_CLOSED_FLUX_CU_

#include "ZQ_CUDA_PoissonSolver2D_ClosedFlux.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"

namespace ZQ_CUDA_PoissonSolver2D
{
	__global__
	void SolveFlux_ClosedFlux_u_RedBlack_Kernel(float* out_du, const float* du, const float* dv, const float* divergence, const float* lambda, const float aug_coeff,
										const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; // x is in [0, width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		
		x = x + 1; // then x is in [1, width-1]

		if(x >= width || y >= height)
			return ;

		int rest = x%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		float coeff = 2.0f,sigma = 0.0f;
		
		sigma -= lambda[y*width+x];
		coeff += aug_coeff;
		sigma += aug_coeff*(du[y*(width+1)+x+1]+dv[(y+1)*width+x]-dv[y*width+x]+divergence[y*width+x]-div_per_volume);
		
		sigma += lambda[y*width+x-1];
		coeff += aug_coeff;
		sigma -= aug_coeff*(-du[y*(width+1)+x-1]+dv[(y+1)*width+x-1]-dv[y*width+x-1]+divergence[y*width+x-1]-div_per_volume);
		
		out_du[y*(width+1)+x] = sigma/coeff;
	}
	
	
	__global__
	void SolveFlux_ClosedFlux_v_RedBlack_Kernel(float* out_dv, const float* du, const float* dv, const float* divergence, const float* lambda, const float aug_coeff,
										const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y; //y is in [0,height-2]

		y = y + 1;	// y is in [1, height-1]
		
		if(x >= width || y >= height)
			return ;

		int rest = y%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		float coeff = 2.0f,sigma = 0.0f;
		sigma -= lambda[y*width+x];
		coeff += aug_coeff;
		sigma += aug_coeff*(du[y*(width+1)+x+1]-du[y*(width+1)+x]+dv[(y+1)*width+x]+divergence[y*width+x]-div_per_volume);
		sigma += lambda[(y-1)*width+x];
		coeff += aug_coeff;
		sigma -= aug_coeff*(du[(y-1)*(width+1)+x+1]-du[(y-1)*(width+1)+x]-dv[(y-1)*width+x]+divergence[(y-1)*width+x] - div_per_volume);
		
		out_dv[y*width+x] = sigma/coeff;
	}
	
	__global__
	void SolveFlux_ClosedFlux_occupy_u_RedBlack_Kernel(float* out_du, const float* du, const float* dv, const bool* occupy, const float* divergence, const float* lambda, const float aug_coeff,
										const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; //x is in[0, width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		x = x + 1; // then x s in [1,width-1]
		if(x >= width || y >= height)
			return ;

		int rest = x%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		float coeff = 2.0f,sigma = 0.0f;
		
		if(occupy[y*width+x])
			return ;
		sigma -= lambda[y*width+x];
		coeff += aug_coeff;
		sigma += aug_coeff*(du[y*(width+1)+x+1]+dv[(y+1)*width+x]-dv[y*width+x]+divergence[y*width+x] - div_per_volume);
		
		if(occupy[y*width+x-1])
			return ;
		sigma += lambda[y*width+x-1];
		coeff += aug_coeff;
		sigma -= aug_coeff*(-du[y*(width+1)+x-1]+dv[(y+1)*width+x-1]-dv[y*width+x-1]+divergence[y*width+x-1] - div_per_volume);
		
		out_du[y*(width+1)+x] = sigma/coeff;
	}
	
	
	__global__
	void SolveFlux_ClosedFlux_occupy_v_RedBlack_Kernel(float* out_dv, const float* du, const float* dv, const bool* occupy, const float* divergence, const float* lambda, const float aug_coeff,
										const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y; // y is in [0, height-2]

		y = y + 1; // y is in [1,height-1]
		if(x >= width || y >= height)
			return ;

		int rest = y%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		float coeff = 2.0f,sigma = 0.0f;
		
		if(occupy[y*width+x])
			return ;
		sigma -= lambda[y*width+x];
		coeff += aug_coeff;
		sigma += aug_coeff*(du[y*(width+1)+x+1]-du[y*(width+1)+x]+dv[(y+1)*width+x]+divergence[y*width+x]-div_per_volume);
		
		if(occupy[(y-1)*width+x])
			return ;
		sigma += lambda[(y-1)*width+x];
		coeff += aug_coeff;
		sigma -= aug_coeff*(du[(y-1)*(width+1)+x+1]-du[(y-1)*(width+1)+x]-dv[(y-1)*width+x]+divergence[(y-1)*width+x]-div_per_volume);
		
		out_dv[y*width+x] = sigma/coeff;
	}
	
	__global__
	void SolveFlux_ClosedFlux_FaceRatio_u_RedBlack_Kernel(float* out_du, const float* du, const float* dv, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
										const float* divergence, const float* lambda, const float aug_coeff, 
										const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; // x is in [0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		x = x + 1; // x is in [1,width-1]
		if(x >= width || y >= height)
			return ;

		int rest = x%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		float ratio = unoccupyU[y*(width+1)+x];
		float ratio2 = ratio*ratio;
		
		if(ratio == 0)
			return ;
		
		float coeff = 2.0f*ratio,sigma = 0.0f;
		
		sigma -= ratio*lambda[y*width+x];
		coeff += ratio2*aug_coeff;
		sigma += ratio*aug_coeff*(
					unoccupyU[y*(width+1)+x+1]*du[y*(width+1)+x+1]
				   +unoccupyV[(y+1)*width+x]*dv[(y+1)*width+x]
				   -unoccupyV[y*width+x]*dv[y*width+x]
				   +divergence[y*width+x]-div_per_volume*unoccupyVolume[y*width+x]);
		
		sigma += ratio*lambda[y*width+x-1];
		coeff += ratio2*aug_coeff;
		sigma -= ratio*aug_coeff*(
					-unoccupyU[y*(width+1)+x-1]*du[y*(width+1)+x-1]
					+unoccupyV[(y+1)*width+x-1]*dv[(y+1)*width+x-1]
					-unoccupyV[y*width+x-1]*dv[y*width+x-1]
					+divergence[y*width+x-1]-div_per_volume*unoccupyVolume[y*width+x-1]);
		out_du[y*(width+1)+x] = sigma/coeff;
	}
	
	__global__
	void SolveFlux_ClosedFlux_FaceRatio_v_RedBlack_Kernel(float* out_dv, const float* du, const float* dv, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
										const float* divergence, const float* lambda, const float aug_coeff, 
										const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y; // y is in [0,height-2]

		y = y + 1; //then y is in [1,height-1]
		if(x >= width || y >= height)
			return ;

		int rest = y%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		float ratio = unoccupyV[y*width+x];
		
		if(ratio == 0)
			return ;
			
		float ratio2 = ratio*ratio;
		float coeff = 2.0f*ratio,sigma = 0.0f;
		
		sigma -= ratio*lambda[y*width+x];
		coeff += ratio2*aug_coeff;
		sigma += ratio*aug_coeff*(
					unoccupyU[y*(width+1)+x+1]*du[y*(width+1)+x+1]
				   -unoccupyU[y*(width+1)+x]*du[y*(width+1)+x]
				   +unoccupyV[(y+1)*width+x]*dv[(y+1)*width+x]
				   +divergence[y*width+x]-div_per_volume*unoccupyVolume[y*width+x]);
		
		sigma += ratio*lambda[(y-1)*width+x];
		coeff += ratio2*aug_coeff;
		sigma -= ratio*aug_coeff*(
					unoccupyU[(y-1)*(width+1)+x+1]*du[(y-1)*(width+1)+x+1]
				   -unoccupyU[(y-1)*(width+1)+x]*du[(y-1)*(width+1)+x]
				   -unoccupyV[(y-1)*width+x]*dv[(y-1)*width+x]
				   +divergence[(y-1)*width+x]-div_per_volume*unoccupyVolume[(y-1)*width+x]);
		
		out_dv[y*width+x] = sigma/coeff;
	}
	
	/********************************************************/
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveClosedFluxRedBlack_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* tmp_div_d = 0;
		float* lambda_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&lambda_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&tmp_div_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(lambda_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(tmp_div_d,0,sizeof(float)*width*height));
		
		float* du_d = 0;
		float* dv_d = 0;
		float* tmp_du_d = 0;
		float* tmp_dv_d = 0;
		checkCudaErrors( cudaMalloc((void**)&du_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&dv_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&tmp_du_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dv_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(du_d,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(dv_d,0,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(tmp_du_d,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(tmp_dv_d,0,sizeof(float)*width*(height+1)) );
		

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);
		
		float aug_coeff = 1.0f;
		for(int out_it = 0; out_it < outerIter; out_it++)
		{
			//Red-Black Solve du,dv
			for(int rd_it = 0; rd_it < innerIter; rd_it++)
			{
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,b_d,lambda_d,aug_coeff,div_per_volume,width,height,true);
				
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,b_d,lambda_d,aug_coeff,div_per_volume,width,height,false);			
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,b_d,lambda_d,aug_coeff,div_per_volume,width,height,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,b_d,lambda_d,aug_coeff,div_per_volume,width,height,false);
			}
			
			Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(tmp_div_d,du_d,dv_d,width,height);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,b_d,1.0f,width,height,1);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(lambda_d,tmp_div_d,-aug_coeff,width,height,1);
			
			aug_coeff *= 2.0f;
		}
		
		dim3 uu_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 vv_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<uu_gridSize,blockSize>>>(mac_u,du_d,1.0f,width+1,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<vv_gridSize,blockSize>>>(mac_v,dv_d,1.0f,width,height+1,1);
		
		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(tmp_div_d) );
		checkCudaErrors( cudaFree(lambda_d) );
		checkCudaErrors( cudaFree(du_d) );
		checkCudaErrors( cudaFree(dv_d) );
		checkCudaErrors( cudaFree(tmp_du_d) );
		checkCudaErrors( cudaFree(tmp_dv_d) );
		b_d = 0;
		tmp_div_d = 0;
		lambda_d = 0;
		du_d = 0;
		dv_d = 0;
		tmp_du_d = 0;
		tmp_dv_d = 0;
	}
	
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveClosedFluxRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, const bool* occupy, const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* tmp_div_d = 0;
		float* lambda_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&lambda_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&tmp_div_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(lambda_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(tmp_div_d,0,sizeof(float)*width*height));
		
		float* du_d = 0;
		float* dv_d = 0;
		float* tmp_du_d = 0;
		float* tmp_dv_d = 0;
		checkCudaErrors( cudaMalloc((void**)&du_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&dv_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&tmp_du_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dv_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(du_d,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(dv_d,0,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(tmp_du_d,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(tmp_dv_d,0,sizeof(float)*width*(height+1)) );
		

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);
		
		float aug_coeff = 1.0f;
		float max_aug_coeff = 1e6;
		for(int out_it = 0; out_it < outerIter; out_it++)
		{
			//Red-Black Solve du,dv
			for(int rd_it = 0; rd_it < innerIter; rd_it++)
			{
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_occupy_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,occupy,b_d,lambda_d,aug_coeff,div_per_volume,width,height,true);
				
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_occupy_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,occupy,b_d,lambda_d,aug_coeff,div_per_volume,width,height,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_occupy_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,occupy,b_d,lambda_d,aug_coeff,div_per_volume,width,height,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_occupy_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,occupy,b_d,lambda_d,aug_coeff,div_per_volume,width,height,false);
			}
			
			Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(tmp_div_d,du_d,dv_d,width,height);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,b_d,1.0f,width,height,1);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(lambda_d,tmp_div_d,-aug_coeff,width,height,1);
			
			aug_coeff *= 2.0f;
			if(aug_coeff > max_aug_coeff)
				aug_coeff = max_aug_coeff;
		}
		
		dim3 uu_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 vv_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<uu_gridSize,blockSize>>>(mac_u,du_d,1.0f,width+1,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<vv_gridSize,blockSize>>>(mac_v,dv_d,1.0f,width,height+1,1);
		
		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(tmp_div_d) );
		checkCudaErrors( cudaFree(lambda_d) );
		checkCudaErrors( cudaFree(du_d) );
		checkCudaErrors( cudaFree(dv_d) );
		checkCudaErrors( cudaFree(tmp_du_d) );
		checkCudaErrors( cudaFree(tmp_dv_d) );
		b_d = 0;
		tmp_div_d = 0;
		lambda_d = 0;
		du_d = 0;
		dv_d = 0;
		tmp_du_d = 0;
		tmp_dv_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	void cu_SolveClosedFluxRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
										const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* tmp_div_d = 0;
		float* lambda_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&lambda_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&tmp_div_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(lambda_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(tmp_div_d,0,sizeof(float)*width*height));
		
		float* du_d = 0;
		float* dv_d = 0;
		float* tmp_du_d = 0;
		float* tmp_dv_d = 0;
		checkCudaErrors( cudaMalloc((void**)&du_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&dv_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&tmp_du_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_dv_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(du_d,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(dv_d,0,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(tmp_du_d,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(tmp_dv_d,0,sizeof(float)*width*(height+1)) );
		

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);
		
		float aug_coeff = 1.0f;
		float max_aug_coeff = 1e6;
		for(int out_it = 0; out_it < outerIter; out_it++)
		{
			//Red-Black Solve du,dv
			for(int rd_it = 0; rd_it < innerIter; rd_it++)
			{
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_FaceRatio_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,unoccupyVolume,unoccupyU,unoccupyV,b_d,lambda_d,aug_coeff,
																							div_per_volume,width,height,true);
				
				checkCudaErrors( cudaMemcpy(tmp_du_d,du_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_FaceRatio_u_RedBlack_Kernel<<<u_gridSize,blockSize>>>(du_d,tmp_du_d,dv_d,unoccupyVolume,unoccupyU,unoccupyV,b_d,lambda_d,aug_coeff,
																							div_per_volume,width,height,false);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_FaceRatio_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,unoccupyVolume,unoccupyU,unoccupyV,b_d,lambda_d,aug_coeff,
																							div_per_volume,width,height,true);
				
				checkCudaErrors( cudaMemcpy(tmp_dv_d,dv_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToDevice) );
				SolveFlux_ClosedFlux_FaceRatio_v_RedBlack_Kernel<<<v_gridSize,blockSize>>>(dv_d,du_d,tmp_dv_d,unoccupyVolume,unoccupyU,unoccupyV,b_d,lambda_d,aug_coeff,
																							div_per_volume,width,height,false);
			}
			
			Calculate_Divergence_of_MAC_FaceRatio_Kernel<<<gridSize,blockSize>>>(tmp_div_d,du_d,dv_d,unoccupyU,unoccupyV,width,height);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,b_d,1.0f,width,height,1);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_div_d,unoccupyVolume,-div_per_volume,width,height,1);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(lambda_d,tmp_div_d,-aug_coeff,width,height,1);
			
			aug_coeff *= 2.0f;
			if(aug_coeff > max_aug_coeff)
				aug_coeff = max_aug_coeff;
		}
		
		dim3 uu_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 vv_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<uu_gridSize,blockSize>>>(mac_u,du_d,1.0f,width+1,height,1);
		ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<vv_gridSize,blockSize>>>(mac_v,dv_d,1.0f,width,height+1,1);
		
		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(tmp_div_d) );
		checkCudaErrors( cudaFree(lambda_d) );
		checkCudaErrors( cudaFree(du_d) );
		checkCudaErrors( cudaFree(dv_d) );
		checkCudaErrors( cudaFree(tmp_du_d) );
		checkCudaErrors( cudaFree(tmp_dv_d) );
		b_d = 0;
		tmp_div_d = 0;
		lambda_d = 0;
		du_d = 0;
		dv_d = 0;
		tmp_du_d = 0;
		tmp_dv_d = 0;
	}
	
	/*************************************************************/
	
	/*First Implementation*/
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" 
	void SolveClosedFluxRedBlack2D_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );

		cu_SolveClosedFluxRedBlack_MAC(mac_u_d,mac_v_d,div_per_volume,width,height,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		mac_u_d = 0;
		mac_v_d = 0;
	}
	
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" 
	void SolveClosedFluxRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );

		cu_SolveClosedFluxRedBlackwithOccupy_MAC(mac_u_d,mac_v_d,occupy_d,div_per_volume,width,height,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		occupy_d = 0;
	}
	
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" 
	void SolveClosedFluxRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
											const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* unoccupyVolume_d = 0;
		float* unoccupyU_d = 0;
		float* unoccupyV_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyVolume_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyU_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyV_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyVolume_d,unoccupyVolume,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyU_d,unoccupyU,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyV_d,unoccupyV,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		
		cu_SolveClosedFluxRedBlackwithFaceRatio_MAC(mac_u_d,mac_v_d,unoccupyVolume_d,unoccupyU_d,unoccupyV_d,div_per_volume,width,height,outerIter,innerIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(unoccupyVolume_d) );
		checkCudaErrors( cudaFree(unoccupyU_d) );
		checkCudaErrors( cudaFree(unoccupyV_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		unoccupyVolume_d = 0;
		unoccupyU_d = 0;
		unoccupyV_d = 0;
	}
}

#endif