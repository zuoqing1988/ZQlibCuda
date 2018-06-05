#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_CU_
#define _ZQ_CUDA_POISSON_SOLVER_2D_CU_

#include "ZQ_CUDA_PoissonSolver2D.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"

namespace ZQ_CUDA_PoissonSolver2D
{
	__global__
	void Regular_to_MAC_u_Kernel(float* mac_u, const float* u, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		if(x == 0)
			mac_u[y*(width+1)+x] = u[y*width+x];
		else if(x == width)
			mac_u[y*(width+1)+x] = u[y*width+x-1];
		else
			mac_u[y*(width+1)+x] = 0.5f*(u[y*width+x-1]+u[y*width+x]);
	}

	__global__
	void Regular_to_MAC_v_Kernel(float* mac_v, const float* v, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		if(y == 0)
			mac_v[y*width+x] = v[y*width+x];
		else if(y == height)
			mac_v[y*width+x] = v[(y-1)*width+x];
		else
			mac_v[y*width+x] = 0.5f*(v[(y-1)*width+x]+v[y*width+x]);
	}

	__global__
	void MAC_to_Regular_vel_Kernel(float* u, float* v, const float* mac_u, const float* mac_v, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		u[y*width+x] = 0.5f*(mac_u[y*(width+1)+x]+mac_u[y*(width+1)+x+1]);
		v[y*width+x] = 0.5f*(mac_v[y*width+x]+mac_v[(y+1)*width+x]);
	}

	__global__
	void Calculate_Divergence_of_MAC_Kernel(float* divergence, const float* mac_u, const float* mac_v, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		divergence[y*width+x] = mac_u[y*(width+1)+x+1] - mac_u[y*(width+1)+x] + mac_v[(y+1)*width+x] - mac_v[y*width+x];
	}
	
	__global__
	void Calculate_Divergence_of_MAC_FaceRatio_Kernel(float* divergence, const float* mac_u, const float* mac_v, const float* unoccupyU, const float* unoccupyV,
								const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		divergence[y*width+x] = mac_u[y*(width+1)+x+1]*unoccupyU[y*(width+1)+x+1] 
							  - mac_u[y*(width+1)+x]*unoccupyU[y*(width+1)+x] 
							  + mac_v[(y+1)*width+x]*unoccupyV[(y+1)*width+x]
							  - mac_v[y*width+x]*unoccupyV[y*width+x];
	}

	__global__
	void Adjust_MAC_u_OpenPoisson_Kernel(float* mac_u, const float* p, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		if(x == 0)
			mac_u[y*(width+1)+x] -= p[y*width+x] - 0;
		else if(x == width)
			mac_u[y*(width+1)+x] -= 0 - p[y*width+x-1];
		else
			mac_u[y*(width+1)+x] -= p[y*width+x] - p[y*width+x-1];
	}

	__global__
	void Adjust_MAC_v_OpenPoisson_Kernel(float* mac_v, const float* p, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		if(y == 0)
			mac_v[y*width+x] -= p[y*width+x] - 0;
		else if(y == height)
			mac_v[y*width+x] -= 0 - p[(y-1)*width+x];
		else
			mac_v[y*width+x] -= p[y*width+x] - p[(y-1)*width+x];
	}

	__global__
	void Adjust_MAC_u_OpenPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		if(x == 0)
		{
			if(!occupy[y*width+x])
				mac_u[y*(width+1)+x] -= p[y*width+x] - 0;
		}
		else if(x == width)
		{
			if(!occupy[y*width+x-1])
				mac_u[y*(width+1)+x] -= 0 - p[y*width+x-1];
		}
		else
		{
			if(!occupy[y*width+x-1] && !occupy[y*width+x])
				mac_u[y*(width+1)+x] -= p[y*width+x] - p[y*width+x-1];
		}
	}

	__global__
	void Adjust_MAC_v_OpenPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		if(y == 0)
		{
			if(!occupy[y*width+x])
				mac_v[y*width+x] -= p[y*width+x] - 0;
		}
		else if(y == height)
		{
			if(!occupy[(y-1)*width+x])
				mac_v[y*width+x] -= 0 - p[(y-1)*width+x];
		}
		else
		{
			if(!occupy[(y-1)*width+x] && !occupy[y*width+x])
				mac_v[y*width+x] -= p[y*width+x] - p[(y-1)*width+x];
		}
	}

	__global__
	void Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		if(x == 0)
		{
			if(unoccupyU[y*(width+1)+x] != 0)
				mac_u[y*(width+1)+x] -= p[y*width+x] - 0;
		}
		else if(x == width)
		{
			if(unoccupyU[y*(width+1)+x] != 0)
				mac_u[y*(width+1)+x] -= 0 - p[y*width+x-1];
		}
		else
		{
			if(unoccupyU[y*(width+1)+x] != 0)
				mac_u[y*(width+1)+x] -= p[y*width+x] - p[y*width+x-1];
		}
	}
	
	__global__
	void Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		if(y == 0)
		{
			if(unoccupyV[y*width+x] != 0)
				mac_v[y*width+x] -= p[y*width+x] - 0;
		}
		else if(y == height)
		{
			if(unoccupyV[y*width+x] != 0)
				mac_v[y*width+x] -= 0 - p[(y-1)*width+x];
		}
		else
		{
			if(unoccupyV[y*width+x] != 0)
				mac_v[y*width+x] -= p[y*width+x] - p[(y-1)*width+x];
		}
	}
	
	/*First Implementation*/
	__global__
	void SolvePressure_OpenPoisson_RedBlack_Kernel(float* p, const float* divergence, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		int offset = y*width+x;
		float coeff = 0,sigma = 0;

		coeff = 4;

		if(y == 0)
		{
			sigma += p[offset+width];
		}
		else if(y == height-1)
		{
			sigma += p[offset-width];
		}
		else 
		{
			sigma += p[offset-width]+p[offset+width];
		}

		if(x == 0)
		{
			sigma += p[offset+1];
		}
		else if(x == width-1)
		{
			sigma += p[offset-1];
		}
		else
		{
			sigma += p[offset+1]+p[offset-1];
		}

		sigma -= divergence[offset];

		p[offset] = sigma/coeff;
	}

	__global__
	void SolvePressure_OpenPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		int offset = y*width+x;
		float coeff = 0,sigma = 0;

		if(occupy[offset])
		{
			p[offset] = 0;
			return ;
		}

		coeff = 4;

		if(y == 0)
		{
			if(!occupy[offset+width])
				sigma += p[offset+width];
			else
				coeff -= 1;
		}
		else if(y == height-1)
		{
			if(!occupy[offset-width])
				sigma += p[offset-width];
			else
				coeff -= 1;
		}
		else 
		{
			if(!occupy[offset+width])
				sigma += p[offset+width];
			else
				coeff -= 1;
			if(!occupy[offset-width])
				sigma += p[offset-width];
			else
				coeff -= 1;
		}

		if(x == 0)
		{
			if(!occupy[offset+1])
				sigma += p[offset+1];
			else
				coeff -= 1;
		}
		else if(x == width-1)
		{
			if(!occupy[offset-1])
				sigma += p[offset-1];
			else
				coeff -= 1;
		}
		else
		{
			if(!occupy[offset+1])
				sigma += p[offset+1];
			else
				coeff -= 1;
			if(!occupy[offset-1])
				sigma += p[offset-1];
			else
				coeff -= 1;
		}

		sigma -= divergence[offset];
		if(coeff > 0)
			p[offset] = sigma/coeff;
		else
			p[offset] = 0;
	}
	
	__global__
	void SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
										const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;

		int offset = y*width+x;
		float coeff = 0,sigma = 0;

		if(occupy[offset])
		{
			p[offset] = 0;
			return ;
		}

		if(y == 0)
		{
			float cur_ratio = unoccupyV[(y+1)*width+x];
			sigma += cur_ratio*p[offset+width];
			coeff += cur_ratio;
			coeff += 1;
		}
		else if(y == height-1)
		{
			float cur_ratio = unoccupyV[y*width+x];
			sigma += cur_ratio*p[offset-width];
			coeff += cur_ratio;
			coeff += 1;
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
			coeff += 1;
		}
		else if(x == width-1)
		{
			float cur_ratio = unoccupyU[y*(width+1)+x];
			sigma += cur_ratio*p[offset-1];
			coeff += cur_ratio;
			coeff += 1;
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

		sigma -= divergence[offset];
		if(coeff > 0)
			p[offset] = sigma/coeff;
		else
			p[offset] = 0;
	}
	
	/*Another Implementation*/
	__global__
	void SolvePressure_OpenPoisson_RedBlack2_Kernel(float* p, const float* divergence, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;

		if(x >= width)
			return ;

		int rest = x%2;

		int start = redkernel ? rest : (1-rest);
		
		for(int y = start;y < height; y+= 2)
		{

			int offset = y*width+x;
			float coeff = 0,sigma = 0;

			coeff = 4;

			if(y == 0)
			{
				sigma += p[offset+width];
			}
			else if(y == height-1)
			{
				sigma += p[offset-width];
			}
			else 
			{
				sigma += p[offset-width]+p[offset+width];
			}

			if(x == 0)
			{
				sigma += p[offset+1];
			}
			else if(x == width-1)
			{
				sigma += p[offset-1];
			}
			else
			{
				sigma += p[offset+1]+p[offset-1];
			}

			sigma -= divergence[offset];

			p[offset] = sigma/coeff;
		}
	}

	__global__
	void SolvePressure_OpenPoisson_occupy_RedBlack2_Kernel(float* p, const float* divergence, const bool* occupy, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;

		if(x >= width)
			return ;

		int rest = x%2;

		int start = redkernel ? rest : (1-rest);
		
		for(int y = start;y < height; y += 2)
		{

			int offset = y*width+x;
			float coeff = 0,sigma = 0;

			if(occupy[offset])
			{
				p[offset] = 0;
				return ;
			}

			coeff = 4;

			if(y == 0)
			{
				if(!occupy[offset+width])
					sigma += p[offset+width];
				else
					coeff -= 1;
			}
			else if(y == height-1)
			{
				if(!occupy[offset-width])
					sigma += p[offset-width];
				else
					coeff -= 1;
			}
			else 
			{
				if(!occupy[offset+width])
					sigma += p[offset+width];
				else
					coeff -= 1;
				if(!occupy[offset-width])
					sigma += p[offset-width];
				else
					coeff -= 1;
			}

			if(x == 0)
			{
				if(!occupy[offset+1])
					sigma += p[offset+1];
				else
					coeff -= 1;
			}
			else if(x == width-1)
			{
				if(occupy[offset-1])
					sigma += p[offset-1];
				else
					coeff -= 1;
			}
			else
			{
				if(!occupy[offset+1])
					sigma += p[offset+1];
				else
					coeff -= 1;
				if(!occupy[offset-1])
					sigma += p[offset-1];
				else
					coeff -= 1;
			}

			sigma -= divergence[offset];
			if(coeff > 0)
				p[offset] = sigma/coeff;
			else
				p[offset] = 0;
		}
	}
	
	__global__
	void SolvePressure_OpenPoisson_FaceRatio_RedBlack2_Kernel(float* p, const float* divergence, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
										const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;

		if(x >= width)
			return ;

		int rest = x%2;
		int start = redkernel ? rest : (1-rest);
		
		for(int y = start;y < height; y += 2)
		{
			int offset = y*width+x;
			float coeff = 0,sigma = 0;

			if(occupy[offset])
			{
				p[offset] = 0;
				return ;
			}

			if(y == 0)
			{
				float cur_ratio = unoccupyV[(y+1)*width+x];
				sigma += cur_ratio*p[offset+width];
				coeff += cur_ratio;
				coeff += 1;
			}
			else if(y == height-1)
			{
				float cur_ratio = unoccupyV[y*width+x];
				sigma += cur_ratio*p[offset-width];
				coeff += cur_ratio;
				coeff += 1;
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
				coeff += 1;
			}
			else if(x == width-1)
			{
				float cur_ratio = unoccupyU[y*(width+1)+x];
				sigma += cur_ratio*p[offset-1];
				coeff += cur_ratio;
				coeff += 1;
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

			sigma -= divergence[offset];
			if(coeff > 0)
				p[offset] = sigma/coeff;
			else
				p[offset] = 0;
		}
	}
	
	
	/**************************************************************/
	
	void cu_Regular_to_MAC_vel(float* mac_u, float* mac_v, const float* u, const float* v, const int width, const int height)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);

		Regular_to_MAC_u_Kernel<<<u_gridSize,blockSize>>>(mac_u,u,width,height);
		Regular_to_MAC_v_Kernel<<<v_gridSize,blockSize>>>(mac_v,v,width,height);
	}

	void cu_MAC_to_Regular_vel(float* u, float* v, const float* mac_u, const float* mac_v, const int width, const int height)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		MAC_to_Regular_vel_Kernel<<<gridSize,blockSize>>>(u,v,mac_u,mac_v,width,height);
	}

	/*First Implementation*/
	void cu_SolveOpenPoissonRedBlack_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_OpenPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,width,height,true);
			SolvePressure_OpenPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,width,height,false);
		}

		Adjust_MAC_u_OpenPoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,width,height);
		Adjust_MAC_v_OpenPoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}

	void cu_SolveOpenPoissonRedBlack_Regular(float* u, float* v, const int width, const int height, const int maxIter)
	{
		float* mac_u = 0;
		float* mac_v = 0;
		checkCudaErrors( cudaMalloc((void**)&mac_u,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(mac_u,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(mac_v,0,sizeof(float)*width*(height+1)) );

		cu_Regular_to_MAC_vel(mac_u,mac_v,u,v,width,height);
		cu_SolveOpenPoissonRedBlack_MAC(mac_u,mac_v,width,height,maxIter);
		cu_MAC_to_Regular_vel(u,v,mac_u,mac_v,width,height);

		checkCudaErrors( cudaFree(mac_u) );
		checkCudaErrors( cudaFree(mac_v) );
		mac_u = 0;
		mac_v = 0;
	}

	void cu_SolveOpenPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_OpenPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,width,height,true);
			SolvePressure_OpenPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,width,height,false);
		}

		Adjust_MAC_u_OpenPoisson_occupy_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,occupy,width,height);
		Adjust_MAC_v_OpenPoisson_occupy_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,occupy,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveOpenPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
													const int width ,const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,unoccupyU,unoccupyV,width,height,true);
			SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,unoccupyU,unoccupyV,width,height,false);
		}

		Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,unoccupyU,width,height);
		Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,unoccupyV,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	/*Another Implementation*/
	void cu_SolveOpenPoissonRedBlack2_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		dim3 len_blockSize(16,1);
		dim3 len_gridSize((width+len_blockSize.x-1)/len_blockSize.x, 1);
		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_OpenPoisson_RedBlack2_Kernel<<<len_gridSize,len_blockSize>>>(p_d,b_d,width,height,true);
			SolvePressure_OpenPoisson_RedBlack2_Kernel<<<len_gridSize,len_blockSize>>>(p_d,b_d,width,height,false);
		}

		Adjust_MAC_u_OpenPoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,width,height);
		Adjust_MAC_v_OpenPoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}

	void cu_SolveOpenPoissonRedBlack2_Regular(float* u, float* v, const int width, const int height, const int maxIter)
	{
		float* mac_u = 0;
		float* mac_v = 0;
		checkCudaErrors( cudaMalloc((void**)&mac_u,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemset(mac_u,0,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMemset(mac_v,0,sizeof(float)*width*(height+1)) );

		cu_Regular_to_MAC_vel(mac_u,mac_v,u,v,width,height);
		cu_SolveOpenPoissonRedBlack2_MAC(mac_u,mac_v,width,height,maxIter);
		cu_MAC_to_Regular_vel(u,v,mac_u,mac_v,width,height);

		checkCudaErrors( cudaFree(mac_u) );
		checkCudaErrors( cudaFree(mac_v) );
		mac_u = 0;
		mac_v = 0;
	}

	void cu_SolveOpenPoissonRedBlackwithOccupy2_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		dim3 len_blockSize(16,1);
		dim3 len_gridSize((width+len_blockSize.x-1)/len_blockSize.x, 1);
		
		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_OpenPoisson_occupy_RedBlack2_Kernel<<<len_gridSize,len_blockSize>>>(p_d,b_d,occupy,width,height,true);
			SolvePressure_OpenPoisson_occupy_RedBlack2_Kernel<<<len_gridSize,len_blockSize>>>(p_d,b_d,occupy,width,height,false);
		}

		Adjust_MAC_u_OpenPoisson_occupy_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,occupy,width,height);
		Adjust_MAC_v_OpenPoisson_occupy_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,occupy,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveOpenPoissonRedBlackwithFaceRatio2_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV, 
													const int width ,const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);


		float* b_d = 0;
		float* p_d = 0;
		checkCudaErrors( cudaMalloc((void**)&b_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMalloc((void**)&p_d,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(b_d,0,sizeof(float)*width*height));
		checkCudaErrors( cudaMemset(p_d,0,sizeof(float)*width*height));

		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(b_d,mac_u,mac_v,width,height);

		dim3 len_blockSize(16,1);
		dim3 len_gridSize((width+len_blockSize.x-1)/len_blockSize.x, 1);
		for(int i = 0;i < maxIter;i++)
		{
			SolvePressure_OpenPoisson_FaceRatio_RedBlack2_Kernel<<<len_gridSize,len_blockSize>>>(p_d,b_d,occupy,unoccupyU,unoccupyV,width,height,true);
			SolvePressure_OpenPoisson_FaceRatio_RedBlack2_Kernel<<<len_gridSize,len_blockSize>>>(p_d,b_d,occupy,unoccupyU,unoccupyV,width,height,false);
		}

		Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,unoccupyU,width,height);
		Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,unoccupyV,width,height);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	
	
	
	/*****************************************************************************/

	/*First Implementation*/
	extern "C" 
	void SolveOpenPoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlack_MAC(mac_u_d,mac_v_d,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		mac_u_d = 0;
		mac_v_d = 0;
	}

	extern "C"
	void SolveOpenPoissonRedBlack2D_Regular(float* u, float* v, const int width, const int height, const int maxIter)
	{
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlack_Regular(u_d,v_d,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		u_d = 0;
		v_d = 0;
	}

	extern "C" 
	void SolveOpenPoissonRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter)
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

		cu_SolveOpenPoissonRedBlackwithOccupy_MAC(mac_u_d,mac_v_d,occupy_d,width,height,maxIter);

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
	void SolveOpenPoissonRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV,
										const int width, const int height, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* occupy_d = 0;
		float* unoccupyU_d = 0;
		float* unoccupyV_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(bool)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyU_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyV_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyU_d,unoccupyU,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyV_d,unoccupyV,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlackwithFaceRatio_MAC(mac_u_d,mac_v_d,occupy_d,unoccupyU_d,unoccupyV_d,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(unoccupyU_d) );
		checkCudaErrors( cudaFree(unoccupyV_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		occupy_d = 0;
		unoccupyU_d = 0;
		unoccupyV_d = 0;
	}
	
	/*Another Implementation*/
	extern "C" 
	void SolveOpenPoissonRedBlack2_2D_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlack2_MAC(mac_u_d,mac_v_d,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		mac_u_d = 0;
		mac_v_d = 0;
	}

	extern "C"
	void SolveOpenPoissonRedBlack2_2D_Regular(float* u, float* v, const int width, const int height, const int maxIter)
	{
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlack2_Regular(u_d,v_d,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		u_d = 0;
		v_d = 0;
	}

	extern "C" 
	void SolveOpenPoissonRedBlackwithOccupy2_2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter)
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

		cu_SolveOpenPoissonRedBlackwithOccupy2_MAC(mac_u_d,mac_v_d,occupy_d,width,height,maxIter);

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
	void SolveOpenPoissonRedBlackwithFaceRatio2_2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV,
										const int width, const int height, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* occupy_d = 0;
		float* unoccupyU_d = 0;
		float* unoccupyV_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(bool)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyU_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyV_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyU_d,unoccupyU,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyV_d,unoccupyV,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlackwithFaceRatio2_MAC(mac_u_d,mac_v_d,occupy_d,unoccupyU_d,unoccupyV_d,width,height,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(unoccupyU_d) );
		checkCudaErrors( cudaFree(unoccupyV_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		occupy_d = 0;
		unoccupyU_d = 0;
		unoccupyV_d = 0;
	}
	
}

#endif