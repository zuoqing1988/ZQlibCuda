#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_CLOSED_POISSON_CU_
#define _ZQ_CUDA_POISSON_SOLVER_2D_CLOSED_POISSON_CU_

#include "ZQ_CUDA_PoissonSolver2D_ClosedPoisson.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"


namespace ZQ_CUDA_PoissonSolver2D
{
	__global__
	void Adjust_MAC_u_ClosedPoisson_Kernel(float* mac_u, const float* p, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;	 //warning: x is in[0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		x = x + 1;	// then x is in [1,width-1]
		if(x >= width || y >= height)
			return ;
			
		mac_u[y*(width+1)+x] -= p[y*width+x] - p[y*width+x-1];
	}

	__global__
	void Adjust_MAC_v_ClosedPoisson_Kernel(float* mac_v, const float* p, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;		// warning: y is in [0, height-2]
		
		y = y + 1;	// then y is in [1,height-1]

		if(x >= width || y >= height)
			return ;

		mac_v[y*width+x] -= p[y*width+x] - p[(y-1)*width+x];
	}

	__global__
	void Adjust_MAC_u_ClosedPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; //warning: x is in[0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		x = x + 1;	// then x is in [1,width-1]
		
		if(x >= width || y >= height)
			return ;
			
		if(!occupy[y*width+x-1] && !occupy[y*width+x])
			mac_u[y*(width+1)+x] -= p[y*width+x] - p[y*width+x-1];
	}

	__global__
	void Adjust_MAC_v_ClosedPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;	// warning: y is in [0, height-2]

		y = y + 1;	// then y is in [1,height-1]
		
		if(x >= width || y >= height)
			return ;
	
		
		if(!occupy[(y-1)*width+x] && !occupy[y*width+x])
			mac_v[y*width+x] -= p[y*width+x] - p[(y-1)*width+x];
	}
	
	__global__
	void Adjust_MAC_u_ClosedPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; //warning: x is in[0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y; 

		x = x + 1; // then x is in [1,width-1]
		
		if(x >= width || y >= height)
			return ;

		if(unoccupyU[y*(width+1)+x] != 0)
			mac_u[y*(width+1)+x] -= p[y*width+x] - p[y*width+x-1];
	}
	
	__global__
	void Adjust_MAC_v_ClosedPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;	// warning: y is in [0, height-2]

		y = y + 1;	// then y is in [1,height-1]
		
		if(x >= width || y >= height)
			return ;
		
		if(unoccupyV[y*width+x] != 0)
			mac_v[y*width+x] -= p[y*width+x] - p[(y-1)*width+x];
	}
	
	__global__
	void SolvePressure_ClosedPoisson_RedBlack_Kernel(float* p, const float* divergence, const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
		
		//if(x == 0 && y == 0)	//the first one is set zero
		//	return ;

		int offset = y*width+x;
		float coeff = 0;
		float sigma = 0;
		
		if(y == 0)
		{
			sigma += p[offset+width];
			coeff += 1.0f;
		}
		else if(y == height-1)
		{
			sigma += p[offset-width];
			coeff += 1.0f;
		}
		else 
		{
			sigma += p[offset-width]+p[offset+width];
			coeff += 2.0f;
		}

		if(x == 0)
		{
			sigma += p[offset+1];
			coeff += 1.0f;
		}
		else if(x == width-1)
		{
			sigma += p[offset-1];
			coeff += 1.0f;
		}
		else
		{
			sigma += p[offset+1]+p[offset-1];
			coeff += 2.0f;
		}

		sigma -= divergence[offset] - div_per_volume;

		p[offset] = sigma/coeff;
	}
	
	__global__
	void SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, 
										/*const int first_x, const int first_y,*/ const float div_per_volume, 
										const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
			
		//if(x == first_x && y == first_y)
		//	return;

		int offset = y*width+x;
		

		if(occupy[offset])
		{
			p[offset] = 0;
			return ;
		}
		float coeff = 0.0f,sigma = 0.0f;

		if(y == 0)
		{
			if(!occupy[offset+width])
			{
				sigma += p[offset+width];
				coeff += 1.0f;
			}
			
		}
		else if(y == height-1)
		{
			if(!occupy[offset-width])
			{
				sigma += p[offset-width];
				coeff += 1.0f;
			}
		}
		else 
		{
			if(!occupy[offset+width])
			{
				sigma += p[offset+width];
				coeff += 1.0f;
			}
			
			if(!occupy[offset-width])
			{
				sigma += p[offset-width];
				coeff += 1.0f;
			}
		}

		if(x == 0)
		{
			if(!occupy[offset+1])
			{
				sigma += p[offset+1];
				coeff += 1.0f;
			}	
		}
		else if(x == width-1)
		{
			if(!occupy[offset-1])
			{
				sigma += p[offset-1];
				coeff += 1.0f;
			}
		}
		else
		{
			if(!occupy[offset+1])
			{
				sigma += p[offset+1];
				coeff += 1.0f;
			}
			if(!occupy[offset-1])
			{
				sigma += p[offset-1];
				coeff += 1.0f;
			}
		}

		sigma -= divergence[offset] - div_per_volume;
		
		p[offset] = sigma/coeff;
		
	}
	
	__global__
	void SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, 
										/*const int first_x, const int first_y,*/ const float div_per_volume, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
			
		//if(x == first_x && y == first_y)
		//	return ;

		int offset = y*width+x;
		float coeff = 0,sigma = 0;

		if(unoccupyVolume[offset] == 0)
		{
			p[offset] = 0;
			return ;
		}

		if(y == 0)
		{
			float cur_ratio = unoccupyV[(y+1)*width+x];
			sigma += cur_ratio*p[offset+width];
			coeff += cur_ratio;
		}
		else if(y == height-1)
		{
			float cur_ratio = unoccupyV[y*width+x];
			sigma += cur_ratio*p[offset-width];
			coeff += cur_ratio;
		}
		else 
		{
			float cur_ratio = unoccupyV[(y+1)*width+x];
			sigma += cur_ratio*p[offset+width];
			coeff += cur_ratio;
			cur_ratio = unoccupyV[y*width+x];
			sigma += cur_ratio*p[offset-width];
			coeff += cur_ratio;
		}

		if(x == 0)
		{
			float cur_ratio = unoccupyU[y*(width+1)+x+1];
			sigma += cur_ratio*p[offset+1];
			coeff += cur_ratio;
		}
		else if(x == width-1)
		{
			float cur_ratio = unoccupyU[y*(width+1)+x];
			sigma += cur_ratio*p[offset-1];
			coeff += cur_ratio;
		}
		else
		{
			float cur_ratio = unoccupyU[y*(width+1)+x+1];
			sigma += cur_ratio*p[offset+1];
			coeff += cur_ratio;
			cur_ratio = unoccupyU[y*(width+1)+x];
			sigma += cur_ratio*p[offset-1];
			coeff += cur_ratio;
		}

		sigma -= divergence[offset] - div_per_volume*unoccupyVolume[offset];
		if(coeff > 0)
			p[offset] = sigma/coeff;
		else
			p[offset] = 0;
	}

	/*****************************************************************************/
	void cu_SolveClosedPoissonRedBlack_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_ClosedPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,div_per_volume,width,height,true);
			SolvePressure_ClosedPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,div_per_volume,width,height,false);
		}

		Adjust_MAC_u_ClosedPoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,width,height);
		Adjust_MAC_v_ClosedPoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveClosedPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, const bool* occupy, /* const int first_x, const int first_y,*/ const float div_per_volume,
										 const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,/*first_x,first_y,*/div_per_volume,width,height,true);
			SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,/*first_x,first_y,*/div_per_volume,width,height,false);
		}

		Adjust_MAC_u_ClosedPoisson_occupy_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,occupy,width,height);
		Adjust_MAC_v_ClosedPoisson_occupy_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,occupy,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveClosedPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, 
									/*const int first_x, const int first_y,*/ const float div_per_volume, const int width ,const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,unoccupyVolume,unoccupyU,unoccupyV,/*first_x,first_y,*/div_per_volume,width,height,true);
			SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,unoccupyVolume,unoccupyU,unoccupyV,/*first_x,first_y,*/div_per_volume,width,height,false);
		}

		Adjust_MAC_u_ClosedPoisson_FaceRatio_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,unoccupyU,width,height);
		Adjust_MAC_v_ClosedPoisson_FaceRatio_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,unoccupyV,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	
	/*************************************************************/
	extern "C" 
	void SolveClosedPoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );

		cu_SolveClosedPoissonRedBlack_MAC(mac_u_d,mac_v_d,div_per_volume,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		mac_u_d = 0;
		mac_v_d = 0;
	}


	extern "C" 
	void SolveClosedPoissonRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy, /*const int first_x, const int first_y,*/ const float div_per_volume,
					 const int width, const int height, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(bool)*width*height) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );

		cu_SolveClosedPoissonRedBlackwithOccupy_MAC(mac_u_d,mac_v_d,occupy_d,/*first_x,first_y,*/div_per_volume,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		occupy_d = 0;
	}
	
	extern "C" 
	void SolveClosedPoissonRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
										/*const int first_x, const int first_y,*/ const float div_per_volume, const int width, const int height, const int maxIter)
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

		cu_SolveClosedPoissonRedBlackwithFaceRatio_MAC(mac_u_d,mac_v_d,unoccupyVolume_d,unoccupyU_d,unoccupyV_d,/*first_x,first_y,*/div_per_volume,width,height,maxIter);

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