#ifndef _ZQ_CUDA_POISSON_SOLVER_3D_CU_
#define _ZQ_CUDA_POISSON_SOLVER_3D_CU_

#include "ZQ_CUDA_PoissonSolver3D.cuh"
#include "ZQ_CUDA_ImageProcessing3D.cuh"

namespace ZQ_CUDA_PoissonSolver3D
{
	__global__
	void Regular_to_MAC_u_Kernel(float* mac_u, const float* u, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(x == 0)
				mac_u[z*height*(width+1)+y*(width+1)+x] = u[z*height*width+y*width+x];
			else if(x == width)
				mac_u[z*height*(width+1)+y*(width+1)+x] = u[z*height*width+y*width+x-1];
			else
				mac_u[z*height*(width+1)+y*(width+1)+x] = 0.5f*(u[z*height*width+y*width+x-1]+u[z*height*width+y*width+x]);
		}
	}
	
	__global__
	void Regular_to_MAC_v_Kernel(float* mac_v, const float* v, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(y == 0)
				mac_v[z*(height+1)*width+y*width+x] = v[z*height*width+y*width+x];
			else if(y == height)
				mac_v[z*(height+1)*width+y*width+x] = v[z*height*width+(y-1)*width+x];
			else
				mac_v[z*(height+1)*width+y*width+x] = 0.5f*(v[z*height*width+(y-1)*width+x]+v[z*height*width+y*width+x]);
		}
	}
	
	__global__
	void Regular_to_MAC_w_Kernel(float* mac_w, const float* w, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		mac_w[y*width+x] = w[y*width+x];
		mac_w[depth*height*width+y*width+x] = w[(depth-1)*height*width+x];
		
		for(int z = 1;z < depth;z++)
		{
			mac_w[z*height*width+y*width+x] = 0.5f*(w[(z-1)*height*width+y*width+x]+w[z*height*width+y*width+x]);
		}
	}
	
	__global__
	void MAC_to_Regular_vel_Kernel(float* u, float* v, float* w, const float* mac_u, const float* mac_v, const float* mac_w, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			u[z*height*width+y*width+x] = 0.5f*(mac_u[z*height*(width+1)+y*(width+1)+x]+mac_u[z*height*(width+1)+y*(width+1)+x+1]);
			v[z*height*width+y*width+x] = 0.5f*(mac_v[z*(height+1)*width+y*width+x]+mac_v[z*(height+1)*width+(y+1)*width+x]);
			w[z*height*width+y*width+x] = 0.5f*(mac_w[z*height*width+y*width+x]+mac_w[(z+1)*height*width+y*width+x]);
		}
	}
	
	__global__
	void Calculate_Divergence_of_MAC_Kernel(float* divergence, const float* mac_u, const float* mac_v, const float* mac_w, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			divergence[z*height*width+y*width+x] = mac_u[z*height*(width+1)+y*(width+1)+x+1] - mac_u[z*height*(width+1)+y*(width+1)+x] 
												+ mac_v[z*(height+1)*width+(y+1)*width+x] - mac_v[z*(height+1)*width+y*width+x]
												+ mac_w[(z+1)*height*width+y*width+x] - mac_w[z*height*width+y*width+x];
		}
	}
	
	__global__
	void Calculate_Divergence_of_MAC_FaceRatio_Kernel(float* divergence, const float* mac_u, const float* mac_v, const float* mac_w, 
								const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
								const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			divergence[z*height*width+y*width+x] = 
									mac_u[z*height*(width+1)+y*(width+1)+x+1]*unoccupyU[z*height*(width+1)+y*(width+1)+x+1] 
								  - mac_u[z*height*(width+1)+y*(width+1)+x]*unoccupyU[z*height*(width+1)+y*(width+1)+x] 
								  + mac_v[z*(height+1)*width+(y+1)*width+x]*unoccupyV[z*(height+1)*width+(y+1)*width+x]
								  - mac_v[z*(height+1)*width+y*width+x]*unoccupyV[z*(height+1)*width+y*width+x]
								  + mac_w[(z+1)*height*width+y*width+x]*unoccupyW[(z+1)*height*width+y*width+x]
								  - mac_w[z*height*width+y*width+x]*unoccupyW[z*height*width+y*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_u_OpenPoisson_Kernel(float* mac_u, const float* p, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(x == 0)
				mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - 0;
			else if(x == width)
				mac_u[z*height*(width+1)+y*(width+1)+x] -= 0 - p[z*height*width+y*width+x-1];
			else
				mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - p[z*height*width+y*width+x-1];
		}
	}
	
	__global__
	void Adjust_MAC_v_OpenPoisson_Kernel(float* mac_v, const float* p, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(y == 0)
				mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - 0;
			else if(y == height)
				mac_v[z*(height+1)*width+y*width+x] -= 0 - p[z*height*width+(y-1)*width+x];
			else
				mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - p[z*height*width+(y-1)*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_w_OpenPoisson_Kernel(float* mac_w, const float* p, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		mac_w[y*width+x] -= p[y*width+x] - 0;
		mac_w[depth*height*width+y*width+x] -= 0 - p[(depth-1)*height*width+y*width+x];
		for(int z = 1;z < depth;z++)
		{
			mac_w[z*height*width+y*width+x] -= p[z*height*width+y*width+x] - p[(z-1)*height*width+y*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_u_OpenPoisson_occupy_Kernel(float* mac_u, const float* p, const bool* occupy, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(x == 0)
			{
				if(!occupy[z*height*width+y*width+x])
					mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - 0;
			}
			else if(x == width)
			{
				if(!occupy[z*height*width+y*width+x-1])
					mac_u[z*height*(width+1)+y*(width+1)+x] -= 0 - p[z*height*width+y*width+x-1];
			}
			else
			{
				if(!occupy[z*height*width+y*width+x-1] && !occupy[z*height*width+y*width+x])
					mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - p[z*height*width+y*width+x-1];
			}
		}
	}
	
	__global__
	void Adjust_MAC_v_OpenPoisson_occupy_Kernel(float* mac_v, const float* p, const bool* occupy, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(y == 0)
			{
				if(!occupy[z*height*width+y*width+x])
					mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - 0;
			}
			else if(y == height)
			{
				if(!occupy[z*height*width+(y-1)*width+x])
					mac_v[z*(height+1)*width+y*width+x] -= 0 - p[z*height*width+(y-1)*width+x];
			}
			else
			{
				if(!occupy[z*height*width+(y-1)*width+x] && !occupy[z*height*width+y*width+x])
					mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - p[z*height*width+(y-1)*width+x];
			}
		}
	}
	
	__global__
	void Adjust_MAC_w_OpenPoisson_occupy_Kernel(float* mac_w, const float* p, const bool* occupy, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		if(!occupy[y*width+x])
			mac_w[y*width+x] -= p[y*width+x] - 0;
		if(!occupy[(depth-1)*height*width+y*width+x])
			mac_w[depth*height*width+y*width+x] -= 0 - p[(depth-1)*height*width+y*width+x];
			
		for(int z = 1;z < depth;z++)
		{
			if(!occupy[(z-1)*height*width+y*width+x] && !occupy[z*height*width+y*width+x])
				mac_w[z*height*width+y*width+x] -= p[z*height*width+y*width+x] - p[(z-1)*height*width+y*width+x];
		}
	}
	
	__global__
	void Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel(float* mac_u, const float* p, const float* unoccupyU, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(x == 0)
			{
				if(unoccupyU[z*height*(width+1)+y*(width+1)+x] != 0)
					mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - 0;
			}
			else if(x == width)
			{
				if(unoccupyU[z*height*(width+1)+y*(width+1)+x] != 0)
					mac_u[z*height*(width+1)+y*(width+1)+x] -= 0 - p[z*height*width+y*width+x-1];
			}
			else
			{
				if(unoccupyU[z*height*(width+1)+y*(width+1)+x] != 0)
					mac_u[z*height*(width+1)+y*(width+1)+x] -= p[z*height*width+y*width+x] - p[z*height*width+y*width+x-1];
			}
		}
	}
	
	__global__
	void Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel(float* mac_v, const float* p, const float* unoccupyV, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			if(y == 0)
			{
				if(unoccupyV[z*(height+1)*width+y*width+x] != 0)
					mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - 0;
			}
			else if(y == height)
			{
				if(unoccupyV[z*(height+1)*width+y*width+x] != 0)
					mac_v[z*(height+1)*width+y*width+x] -= 0 - p[z*height*width+(y-1)*width+x];
			}
			else
			{
				if(unoccupyV[z*(height+1)*width+y*width+x] != 0)
					mac_v[z*(height+1)*width+y*width+x] -= p[z*height*width+y*width+x] - p[z*height*width+(y-1)*width+x];
			}
		}
	}
	
	__global__
	void Adjust_MAC_w_OpenPoisson_FaceRatio_Kernel(float* mac_w, const float* p, const float* unoccupyW, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		if(unoccupyW[y*width+x] != 0)
			mac_w[y*width+x] -= p[y*width+x] - 0;
		if(unoccupyW[depth*height*width+y*width+x] != 0)
			mac_w[depth*height*width+y*width+x] -= 0 - p[(depth-1)*height*width+y*width+x];
			
		for(int z = 1;z < depth;z++)
		{
			if(unoccupyW[z*height*width+y*width+x] != 0)
				mac_w[z*height*width+y*width+x] -= p[z*height*width+y*width+x] - p[(z-1)*height*width+y*width+x];
		}
	}
	
	/*First Implementation*/
	__global__
	void SolvePressure_OpenPoisson_RedBlack_Kernel(float* p, const float* divergence, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redkernel ? rest : (1-rest);
		
		for(int z = start; z < depth;z += 2)
		{
			int offset = z*height*width+y*width+x;
			float coeff = 0,sigma = 0;

			coeff = 6;
			
			if(z == 0)
			{
				sigma += p[offset+height*width];
			}
			else if(z == depth-1)
			{
				sigma += p[offset-height*width];
			}
			else
			{
				sigma += p[offset-height*width]+p[offset+height*width];
			}
			
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
	void SolvePressure_OpenPoisson_occupy_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redkernel ? rest : (1-rest);

		for(int z = start;z < depth;z += 2)
		{
			int offset = z*height*width+y*width+x;
			float coeff = 0,sigma = 0;

			if(occupy[offset])
			{
				p[offset] = 0;
				continue ;
			}

			coeff = 6;
			
			if(z == 0)
			{
				if(!occupy[offset+height*width])
					sigma += p[offset+height*width];
				else
					coeff -= 1;
			}
			else if(z == depth-1)
			{
				if(!occupy[offset-height*width])
					sigma += p[offset-height*width];
				else
					coeff -= 1;
			}
			else
			{
				if(!occupy[offset+height*width])
					sigma += p[offset+height*width];
				else
					coeff -= 1;
				if(!occupy[offset-height*width])
					sigma += p[offset-height*width];
				else
					coeff -= 1;
			}
			
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
	}
	
	__global__
	void SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel(float* p, const float* divergence, const bool* occupy, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const int width, const int height, const int depth, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;
		
		int start = redkernel ? rest : (1-rest);

		for(int z = start;z < depth; z += 2)
		{
			int offset = z*height*width+y*width+x;
			float coeff = 0,sigma = 0;

			if(occupy[offset])
			{
				p[offset] = 0;
				continue ;
			}
			
			if(z == 0)
			{
				float cur_ratio = unoccupyW[(z+1)*height*width+y*width+x];
				sigma += cur_ratio*p[offset+height*width];
				coeff += cur_ratio;
				coeff += 1;
			}
			else if(z == depth-1)
			{
				float cur_ratio = unoccupyW[z*height*width+y*width+x];
				sigma += cur_ratio*p[offset-height*width];
				coeff += cur_ratio;
				coeff += 1;
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
				coeff += 1;
			}
			else if(y == height-1)
			{
				float cur_ratio = unoccupyV[z*(height+1)*width+y*width+x];
				sigma += cur_ratio*p[offset-width];
				coeff += cur_ratio;
				coeff += 1;
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
				coeff += 1;
			}
			else if(x == width-1)
			{
				float cur_ratio = unoccupyU[z*height*(width+1)+y*(width+1)+x];
				sigma += cur_ratio*p[offset-1];
				coeff += cur_ratio;
				coeff += 1;
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

			sigma -= divergence[offset];
			if(coeff > 0)
				p[offset] = sigma/coeff;
			else
				p[offset] = 0;
		}
	}
	
	/**************************************************************/
	
	void cu_Regular_to_MAC_vel(float* mac_u, float* mac_v, float* mac_w, const float* u, const float* v, const float* w, const int width, const int height, const int depth)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
		dim3 w_gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		Regular_to_MAC_u_Kernel<<<u_gridSize,blockSize>>>(mac_u,u,width,height,depth);
		Regular_to_MAC_v_Kernel<<<v_gridSize,blockSize>>>(mac_v,v,width,height,depth);
		Regular_to_MAC_w_Kernel<<<w_gridSize,blockSize>>>(mac_w,w,width,height,depth);
	}

	void cu_MAC_to_Regular_vel(float* u, float* v, float* w, const float* mac_u, const float* mac_v, const float* mac_w, const int width, const int height, const int depth)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		MAC_to_Regular_vel_Kernel<<<gridSize,blockSize>>>(u,v,w,mac_u,mac_v,mac_w,width,height,depth);
	}
	
	/*First Implementation*/
	void cu_SolveOpenPoissonRedBlack_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
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
			SolvePressure_OpenPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,width,height,depth,true);
			SolvePressure_OpenPoisson_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,width,height,depth,false);
		}

		Adjust_MAC_u_OpenPoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,width,height,depth);
		Adjust_MAC_v_OpenPoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,width,height,depth);
		Adjust_MAC_w_OpenPoisson_Kernel<<<w_gridSize,blockSize>>>(mac_w,p_d,width,height,depth);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveOpenPoissonRedBlack_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int maxIter)
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
		cu_SolveOpenPoissonRedBlack_MAC(mac_u,mac_v,mac_w,width,height,depth,maxIter);
		cu_MAC_to_Regular_vel(u,v,w,mac_u,mac_v,mac_w,width,height,depth);

		checkCudaErrors( cudaFree(mac_u) );
		checkCudaErrors( cudaFree(mac_v) );
		checkCudaErrors( cudaFree(mac_w) );
		mac_u = 0;
		mac_v = 0;
		mac_w = 0;
	}
	
	void cu_SolveOpenPoissonRedBlackwithOccupy_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
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
			SolvePressure_OpenPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,width,height,depth,true);
			SolvePressure_OpenPoisson_occupy_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,width,height,depth,false);
		}

		Adjust_MAC_u_OpenPoisson_occupy_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,occupy,width,height,depth);
		Adjust_MAC_v_OpenPoisson_occupy_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,occupy,width,height,depth);
		Adjust_MAC_w_OpenPoisson_occupy_Kernel<<<w_gridSize,blockSize>>>(mac_w,p_d,occupy,width,height,depth);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	void cu_SolveOpenPoissonRedBlackwithFaceRatio_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
													const int width ,const int height, const int depth, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x,(height+1+blockSize.y-1)/blockSize.y);
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
			SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,unoccupyU,unoccupyV,unoccupyW,width,height,depth,true);
			SolvePressure_OpenPoisson_FaceRatio_RedBlack_Kernel<<<gridSize,blockSize>>>(p_d,b_d,occupy,unoccupyU,unoccupyV,unoccupyW,width,height,depth,false);
		}

		Adjust_MAC_u_OpenPoisson_FaceRatio_Kernel<<<u_gridSize,blockSize>>>(mac_u,p_d,unoccupyU,width,height,depth);
		Adjust_MAC_v_OpenPoisson_FaceRatio_Kernel<<<v_gridSize,blockSize>>>(mac_v,p_d,unoccupyV,width,height,depth);
		Adjust_MAC_w_OpenPoisson_FaceRatio_Kernel<<<w_gridSize,blockSize>>>(mac_w,p_d,unoccupyW,width,height,depth);

		checkCudaErrors( cudaFree(b_d) );
		checkCudaErrors( cudaFree(p_d) );
		b_d = 0;
		p_d = 0;
	}
	
	/*****************************************************************************/

	/*First Implementation*/
	extern "C" 
	void SolveOpenPoissonRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int maxIter)
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

		cu_SolveOpenPoissonRedBlack_MAC(mac_u_d,mac_v_d,mac_w_d,width,height,depth,maxIter);

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
	void SolveOpenPoissonRedBlack3D_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int maxIter)
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

		cu_SolveOpenPoissonRedBlack_Regular(u_d,v_d,w_d,width,height,depth,maxIter);

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

	extern "C" 
	void SolveOpenPoissonRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;
		bool* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(bool)*width*height*height) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height*depth,cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlackwithOccupy_MAC(mac_u_d,mac_v_d,mac_w_d,occupy_d,width,height,depth,maxIter);

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
	void SolveOpenPoissonRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const int width, const int height, const int depth, const int maxIter)
	{
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		float* mac_w_d = 0;
		bool* occupy_d = 0;
		float* unoccupyU_d = 0;
		float* unoccupyV_d = 0;
		float* unoccupyW_d = 0;

		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&mac_w_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(bool)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyU_d,sizeof(float)*(width+1)*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyV_d,sizeof(float)*width*(height+1)*depth) );
		checkCudaErrors( cudaMalloc((void**)&unoccupyW_d,sizeof(float)*width*height*(depth+1)) );
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_w_d,mac_w,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(bool)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyU_d,unoccupyU,sizeof(float)*(width+1)*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyV_d,unoccupyV,sizeof(float)*width*(height+1)*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(unoccupyW_d,unoccupyW,sizeof(float)*width*height*(depth+1),cudaMemcpyHostToDevice) );

		cu_SolveOpenPoissonRedBlackwithFaceRatio_MAC(mac_u_d,mac_v_d,mac_w_d,occupy_d,unoccupyU_d,unoccupyV_d,unoccupyW_d,width,height,depth,maxIter);

		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1)*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_w,mac_w_d,sizeof(float)*width*height*(depth+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(mac_w_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(unoccupyU_d) );
		checkCudaErrors( cudaFree(unoccupyV_d) );
		checkCudaErrors( cudaFree(unoccupyW_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		mac_w_d = 0;
		occupy_d = 0;
		unoccupyU_d = 0;
		unoccupyV_d = 0;
		unoccupyW_d = 0;
	}
	
	
}

#endif