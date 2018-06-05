#ifndef _ZQ_CUDA_POISSON_SOLVER_3D_CLOSED_POISSON_CU_
#define _ZQ_CUDA_POISSON_SOLVER_3D_CLOSED_POISSON_CU_

#include "ZQ_CUDA_PoissonSolver3D_ClosedPoisson.cuh"
#include "ZQ_CUDA_ImageProcessing3D.cuh"


namespace ZQ_CUDA_PoissonSolver3D
{
	__global__
	void Adjust_MAC_u_ClosedPoisson_Kernel(float* mac_u, const float* p, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;	 //warning: x is in[0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		x = x + 1;	// then x is in [1,width-1]
		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
			mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - p[z*height*width+y*width+x-1];
	}

	__global__
	void Adjust_MAC_v_ClosedPoisson_Kernel(float* mac_v, const float* p, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;		// warning: y is in [0, height-2]
		
		y = y + 1;	// then y is in [1,height-1]

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
			mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - p[z*height*width+(y-1)*width+x];
	}
	
	__global__
	void Adjust_MAC_w_ClosedPoisson_Kernel(float* mac_w, const float* p, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;		
		
		if(x >= width || y >= height)
			return ;

		for(int z = 1;z < depth;z++)
			mac_w[z*height*width+y*width+x] -= p[z*height*width+y*width+x] - p[(z-1)*height*width+y*width+x];
	}

	__global__
	void Adjust_MAC_u_ClosedPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; //warning: x is in[0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		x = x + 1;	// then x is in [1,width-1]
		
		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{
			if(!occupy[z*height*width+y*width+x-1] && !occupy[z*height*width+y*width+x])
				mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - p[z*height*width+y*width+x-1];
		}
	}

	__global__
	void Adjust_MAC_v_ClosedPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;	// warning: y is in [0, height-2]

		y = y + 1;	// then y is in [1,height-1]
		
		if(x >= width || y >= height)
			return ;
	
	
		for(int z = 0;z < depth;z++)
		{	
			if(!occupy[z*height*width+(y-1)*width+x] && !occupy[z*height*width+y*width+x])
				mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - p[z*height*width+(y-1)*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_w_ClosedPoisson_occupy_Kernel(float* mac_w, const float* p, const bool* occupy, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;	
		
		if(x >= width || y >= height)
			return ;
	
	
		for(int z = 1;z < depth;z++)
		{	
			if(!occupy[(z-1)*height*width+y*width+x] && !occupy[z*height*width+y*width+x])
				mac_w[z*height*width+y*width+x] -= p[z*height*width+y*width+x] - p[(z-1)*height*width+y*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_u_ClosedPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x; //warning: x is in[0,width-2]
		int y = threadIdx.y + blockIdx.y * blockDim.y; 

		x = x + 1; // then x is in [1,width-1]
		
		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(unoccupyU[z*height*(width+1)+y*(width+1)+x] != 0)
				mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - p[z*height*width+y*width+x-1];
		}
	}
	
	__global__
	void Adjust_MAC_v_ClosedPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;	// warning: y is in [0, height-2]

		y = y + 1;	// then y is in [1,height-1]
		
		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{
			if(unoccupyV[z*(height+1)*width+y*width+x] != 0)
				mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - p[z*height*width+(y-1)*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_w_ClosedPoisson_FaceRatio_Kernel(float* mac_w, const float* p, const float* unoccupyW, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;	
		
		if(x >= width || y >= height)
			return ;
		
		for(int z = 1;z < depth;z++)
		{
			if(unoccupyW[z*height*width+y*width+x] != 0)
				mac_w[z*height*width+y*width+x] -= p[z*height*width+y*width+x] - p[(z-1)*height*width+y*width+x];
		}
	}
	
	__global__
	void SolvePressure_ClosedPoisson_RedBlack_Kernel(float* p, const float* divergence, const float div_per_volume, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redkernel ? rest : (1-rest);

		for(int z = start; z < depth; z += 2)
		{
			int offset = z*height*width+y*width+x;
			float coeff = 0;
			float sigma = 0;
			if(z == 0)
			{
				sigma += p[offset+height*width];
				coeff += 1.0f;
			}
			else if(z == depth-1)
			{
				sigma += p[offset-height*width];
				coeff += 1.0f;
			}
			else
			{
				sigma += p[offset-height*width]+p[offset+height*width];
				coeff += 2.0f;
			}
			
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
	}
	
	__global__
	void SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, 
										const float div_per_volume, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;
		
		int start = redkernel ? rest : (1-rest);

		for(int z = start; z < depth; z += 2)
		{
			int offset = z*height*width+y*width+x;
			
			if(occupy[offset])
			{
				p[offset] = 0;
				continue ;
			}
			float coeff = 0.0f,sigma = 0.0f;

			if(z == 0)
			{
				if(!occupy[offset+height*width])
				{
					sigma += p[offset+height*width];
					coeff += 1.0f;
				}
			}
			else if(z == depth-1)
			{
				if(!occupy[offset-height*width])
				{
					sigma += p[offset-height*width];
					coeff += 1.0f;
				}
			}
			else
			{
				if(!occupy[offset-height*width])
				{
					sigma += p[offset-height*width];
					coeff += 1.0f;
				}
				if(!occupy[offset+height*width])
				{
					sigma += p[offset+height*width];
					coeff += 1.0f;
				}
			}
			
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
	}
	
	__global__
	void SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, 
										const float* unoccupyW, const float div_per_volume, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redkernel ? rest : (1-rest);
			
		for(int z = start; z < depth; z += 2)
		{
			int offset = z*height*width+y*width+x;
			float coeff = 0,sigma = 0;

			if(unoccupyVolume[offset] == 0)
			{
				p[offset] = 0;
				continue ;
			}
			
			if(z == 0)
			{
				float cur_ratio = unoccupyW[(z+1)*height*width+y*width+x];
				sigma += cur_ratio*p[offset+height*width];
				coeff += cur_ratio;
			}
			else if(z == depth-1)
			{
				float cur_ratio = unoccupyW[z*height*width+y*width+x];
				sigma += cur_ratio*p[offset-height*width];
				coeff += cur_ratio;
			}
			else
			{
				float cur_ratio = unoccupyW[(z+1)*height*width+y*width+x];
				sigma += cur_ratio*p[offset+height*width];
				coeff += cur_ratio;
				cur_ratio = unoccupyW[z*height*width+y*width+x];
				sigma += cur_ratio*p[offset-height*width];
				coeff += cur_ratio;
			}
			
			if(y == 0)
			{
				float cur_ratio = unoccupyV[z*(height+1)*width+(y+1)*width+x];
				sigma += cur_ratio*p[offset+width];
				coeff += cur_ratio;
			}
			else if(y == height-1)
			{
				float cur_ratio = unoccupyV[z*(height+1)*width+y*width+x];
				sigma += cur_ratio*p[offset-width];
				coeff += cur_ratio;
			}
			else 
			{
				float cur_ratio = unoccupyV[z*(height+1)*width+(y+1)*width+x];
				sigma += cur_ratio*p[offset+width];
				coeff += cur_ratio;
				cur_ratio = unoccupyV[z*(height+1)*width+y*width+x];
				sigma += cur_ratio*p[offset-width];
				coeff += cur_ratio;
			}

			if(x == 0)
			{
				float cur_ratio = unoccupyU[z*height*(width+1)+y*(width+1)+x+1];
				sigma += cur_ratio*p[offset+1];
				coeff += cur_ratio;
			}
			else if(x == width-1)
			{
				float cur_ratio = unoccupyU[z*height*(width+1)+y*(width+1)+x];
				sigma += cur_ratio*p[offset-1];
				coeff += cur_ratio;
			}
			else
			{
				float cur_ratio = unoccupyU[z*height*(width+1)+y*(width+1)+x+1];
				sigma += cur_ratio*p[offset+1];
				coeff += cur_ratio;
				cur_ratio = unoccupyU[z*height*(width+1)+y*(width+1)+x];
				sigma += cur_ratio*p[offset-1];
				coeff += cur_ratio;
			}

			sigma -= divergence[offset] - div_per_volume*unoccupyVolume[offset];
			if(coeff > 0)
				p[offset] = sigma/coeff;
			else
				p[offset] = 0;
		}
	}

	/*****************************************************************************/
	void cu_SolveClosedPoissonRedBlack_MAC(float* mac_u, float* mac_v, float* mac_w, const float div_per_volume, const int width, const int height, const int depth, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height*depth));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,mac_w,width,height,depth);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_ClosedPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,div_per_volume,width,height,depth,true);
			SolvePressure_ClosedPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,div_per_volume,width,height,depth,false);
		}

		Adjust_MAC_u_ClosedPoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,width,height,depth);
		Adjust_MAC_v_ClosedPoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,width,height,depth);
		Adjust_MAC_w_ClosedPoisson_Kernel<<<w_gridSize,blockSize>>>(mac_w,p_d,width,height,depth);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveClosedPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float div_per_volume,
										 const int width, const int height, const int depth, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height*depth));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,mac_w,width,height,depth);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,div_per_volume,width,height,depth,true);
			SolvePressure_ClosedPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,div_per_volume,width,height,depth,false);
		}

		Adjust_MAC_u_ClosedPoisson_occupy_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,occupy,width,height,depth);
		Adjust_MAC_v_ClosedPoisson_occupy_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,occupy,width,height,depth);
		Adjust_MAC_w_ClosedPoisson_occupy_Kernel<<<w_gridSize,blockSize>>>(mac_w,p_d,occupy,width,height,depth);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveClosedPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, 
									const float* unoccupyW, const float div_per_volume, const int width ,const int height, const int depth, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width-1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height-1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height*depth));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height*depth));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,mac_w,width,height,depth);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,unoccupyVolume,unoccupyU,unoccupyV,unoccupyW,div_per_volume,width,height,depth,true);
			SolvePressure_ClosedPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,unoccupyVolume,unoccupyU,unoccupyV,unoccupyW,div_per_volume,width,height,depth,false);
		}

		Adjust_MAC_u_ClosedPoisson_FaceRatio_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,unoccupyU,width,height,depth);
		Adjust_MAC_v_ClosedPoisson_FaceRatio_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,unoccupyV,width,height,depth);
		Adjust_MAC_w_ClosedPoisson_FaceRatio_Kernel<<<w_gridSize,blockSize>>>(mac_w,p_d,unoccupyV,width,height,depth);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	
	/*************************************************************/
	extern "C" 
	void SolveClosedPoissonRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float div_per_volume, const int width, const int height, const int depth, const int maxIter)
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

		cu_SolveClosedPoissonRedBlack_MAC(mac_u_d,mac_v_d,mac_w_d,div_per_volume,width,height,depth,maxIter);

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


	extern "C" 
	void SolveClosedPoissonRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float div_per_volume,
					 const int width, const int height, const int depth, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;
		bool* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(bool)*width*height*depth) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height*depth,cudaMemcpyHostToDevice) );

		cu_SolveClosedPoissonRedBlackwithOccupy_MAC(mac_u_d,mac_v_d,mac_w_d,occupy_d,div_per_volume,width,height,depth,maxIter);

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
	
	extern "C" 
	void SolveClosedPoissonRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
										const float* unoccupyW, const float div_per_volume, const int width, const int height, const int depth, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;
		float* unoccupyVolume_d = 0;
		float* unoccupyU_d = 0;
		float* unoccupyV_d = 0;
		float* unoccupyW_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyVolume_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyU_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyV_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyW_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyVolume_d,unoccupyVolume,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyU_d,unoccupyU,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyV_d,unoccupyV,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyW_d,unoccupyW,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );

		cu_SolveClosedPoissonRedBlackwithFaceRatio_MAC(mac_u_d,mac_v_d,mac_w_d,unoccupyVolume_d,unoccupyU_d,unoccupyV_d,unoccupyW_d,div_per_volume,width,height,depth,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_w,mac_w_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(mac_w_d) );
		checkCudaErrors( cudaFree(unoccupyVolume_d) );
		checkCudaErrors( cudaFree(unoccupyU_d) );
		checkCudaErrors( cudaFree(unoccupyV_d) );
		checkCudaErrors( cudaFree(unoccupyW_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		mac_w_d = 0;
		unoccupyVolume_d = 0;
		unoccupyU_d = 0;
		unoccupyV_d = 0;
		unoccupyW_d = 0;
	}
}

#endif