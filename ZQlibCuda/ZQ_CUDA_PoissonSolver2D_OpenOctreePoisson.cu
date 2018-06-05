#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_OPENOCTREEPOISSON_CU_
#define _ZQ_CUDA_POISSON_SOLVER_2D_OPENOCTREEPOISSON_CU_

#include "ZQ_CUDA_PoissonSolver2D_OpenOctreePoisson.cuh"

namespace ZQ_CUDA_PoissonSolver2D
{
	/**********  Open Octree Poisson   ************/
	__global__
	void Adjust_MAC_u_OpenOctreePoisson_Kernel(float* mac_u, const float* p_level0, const float* p_level1, const float* p_level2, const float* p_level3, const bool* leaf0,
														const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > width || y >= height)
			return ;
		
		float len1 = -1, len2 = -1;
		if(x == 0)
		{
			if(leaf0[y*width+x])
			{
				len1 = 4.0f;
				len2 = 0.5f;
				mac_u[y*(width+1)+x] -= (p_level0[(y*width+x)] - 0)/(len1+len2);
			}
			else if(leaf1[y/2*width/2+x/2])
			{
				len1 = 4.0f;
				len2 = 1.0f;
				mac_u[y*(width+1)+x] -= (p_level1[y/2*width/2+x/2] - 0)/(len1+len2);
			}
			else if(leaf2[y/4*width/4+x/4])
			{
				len1 = 4.0f;
				len2 = 2.0f;
				mac_u[y*(width+1)+x] -= (p_level2[y/4*width/4+x/4] - 0)/(len1+len2);
			}
			else if(leaf3[y/8*width/8+x/8])
			{
				len1 = 4.0f;
				len2 = 4.0f;
				mac_u[y*(width+1)+x] -= (p_level3[y/8*width/8+x/8] - 0)/(len1+len2);
			}
		}
		else if(x == width)
		{
			if(leaf0[y*width+x-1])
			{
				len1 = 4.0f;
				len2 = 0.5f;
				mac_u[y*(width+1)+x] -= (0 - p_level0[y*width+x-1])/(len1+len2);
			}
			else if(leaf1[y/2*width/2+x/2-1])
			{
				len1 = 4.0f;
				len2 = 1.0f;
				mac_u[y*(width+1)+x] -= (0 - p_level1[y/2*width/2+x/2-1])/(len1+len2);
			}
			else if(leaf2[y/4*width/4+x/4-1])
			{
				len1 = 4.0f;
				len2 = 2.0f;
				mac_u[y*(width+1)+x] -= (0 - p_level2[y/4*width/4+x/4-1])/(len1+len2);
			}
			else if(leaf3[y/8*width/8+x/8-1])
			{
				len1 = 4.0f;
				len2 = 4.0f;
				mac_u[y*(width+1)+x] -= (0 - p_level3[y/8*width/8+x/8-1])/(len1+len2);
			}
		}
		else
		{
			float p1,p2;
			if(leaf0[y*width+x-1])
			{
				len1 = 0.5f;
				p1 = p_level0[y*width+x-1];
			}
			else if(x%2 == 0 && leaf1[y/2*width/2+x/2-1])
			{
				len1 = 1.0f;
				p1 = p_level1[y/2*width/2+x/2-1];
			}
			else if(x%4 == 0 && leaf2[y/4*width/4+x/4-1])
			{
				len1 = 2.0f;
				p1 = p_level2[y/4*width/4+x/4-1];
			}
			else if(x%8 == 0 && leaf3[y/8*width/8+x/8-1])
			{
				len1 = 4.0f;
				p1 = p_level3[y/8*width/8+x/8-1];
			}
			
			if(len1 < 0)
				return ;
			
			if(leaf0[y*width+x])
			{
				len2 = 0.5f;
				p2 = p_level0[y*width+x];
			}
			else if(x%2 == 0 && leaf1[y/2*width/2+x/2])
			{
				len2 = 1.0f;
				p2 = p_level1[y/2*width/2+x/2];
			}
			else if(x%4 == 0 && leaf2[y/4*width/4+x/4])
			{
				len2 = 2.0f;
				p2 = p_level2[y/4*width/4+x/4];
			}
			else if(x%8 == 0 && leaf3[y/8*width/8+x/8])
			{
				len2 = 4.0f;
				p2 = p_level3[y/8*width/8+x/8];
			}
			
			if(len2 < 0)
				return;
			mac_u[y*(width+1)+x] -= (p2-p1)/(len1+len2);
		}
	}
	
	__global__
	void Adjust_MAC_v_OpenOctreePoisson_Kernel(float* mac_v, const float* p_level0, const float* p_level1, const float* p_level2, const float* p_level3, const bool* leaf0,
														const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y > height)
			return ;
		
		float len1 = -1, len2 = -1;
		if(y == 0)
		{
			if(leaf0[y*width+x])
			{
				len1 = 4.0f;
				len2 = 0.5f;
				mac_v[y*width+x] -= (p_level0[y*width+x] - 0)/(len1+len2);
			}
			else if(leaf1[y/2*width/2+x/2])
			{
				len1 = 4.0f;
				len2 = 1.0f;
				mac_v[y*width+x] -= (p_level1[y/2*width/2+x/2] - 0)/(len1+len2);
			}
			else if(leaf2[y/4*width/4+x/4])
			{
				len1 = 4.0f;
				len2 = 2.0f;
				mac_v[y*width+x] -= (p_level2[y/4*width/4+x/4] - 0)/(len1+len2);
			}
			else if(leaf3[y/8*width/8+x/8])
			{
				len1 = 4.0f;
				len2 = 4.0f;
				mac_v[y*width+x] -= (p_level3[y/8*width/8+x/8] - 0)/(len1+len2);
			}
		}
		else if(y == height)
		{
			if(leaf0[(y-1)*width+x])
			{
				len1 = 4.0f;
				len2 = 0.5f;
				mac_v[y*width+x] -= (0 - p_level0[(y-1)*width+x])/(len1+len2);
			}
			else if(leaf1[(y/2-1)*width/2+x/2])
			{
				len1 = 4.0f;
				len2 = 1.0f;
				mac_v[y*width+x] -= (0 - p_level1[(y/2-1)*width/2+x/2])/(len1+len2);
			}
			else if(leaf2[(y/4-1)*width/4+x/4])
			{
				len1 = 4.0f;
				len2 = 2.0f;
				mac_v[y*width+x] -= (0 - p_level2[(y/4-1)*width/4+x/4])/(len1+len2);
			}
			else if(leaf3[(y/8-1)*width/8+x/8])
			{
				len1 = 4.0f;
				len2 = 4.0f;
				mac_v[y*width+x] -= (0 - p_level3[(y/8-1)*width/8+x/8])/(len1+len2);
			}
		}
		else
		{
			float p1,p2;
			if(leaf0[(y-1)*width+x])
			{
				len1 = 0.5f;
				p1 = p_level0[(y-1)*width+x];
			}
			else if(y%2 == 0 && leaf1[(y/2-1)*width/2+x/2])
			{
				len1 = 1.0f;
				p1 = p_level1[(y/2-1)*width/2+x/2];
			}
			else if(y%4 == 0 && leaf2[(y/4-1)*width/4+x/4])
			{
				len1 = 2.0f;
				p1 = p_level2[(y/4-1)*width/4+x/4];
			}
			else if(y%8 == 0 && leaf3[(y/8-1)*width/8+x/8])
			{
				len1 = 4.0f;
				p1 = p_level3[(y/8-1)*width/8+x/8];
			}
			
			if(len1 < 0)
				return;
				
			if(leaf0[y*width+x])
			{
				len2 = 0.5f;
				p2 = p_level0[y*width+x];
			}
			else if(y%2 == 0 && leaf1[y/2*width/2+x/2])
			{
				len2 = 1.0f;
				p2 = p_level1[y/2*width/2+x/2];
			}
			else if(y%4 == 0 && leaf2[y/4*width/4+x/4])
			{
				len2 = 2.0f;
				p2 = p_level2[y/4*width/4+x/4];
			}
			else if(y%8 == 0 && leaf3[y/8*width/8+x/8])
			{
				len2 = 4.0f;
				p2 = p_level3[y/8*width/8+x/8];
			}
			
			if(len2 < 0)
				return ;	
			
			mac_v[y*width+x] -= (p2-p1)/(len1+len2);
		}
	}
	
	__global__
	void Calculate_Divergence_Octree_from_previous_level_Kernel(float* cur_level_divergence, const float* pre_level_divergence, const int cur_level_width, const int cur_level_height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= cur_level_width || y >= cur_level_height)
			return ;
		
		cur_level_divergence[y*cur_level_width+x] = pre_level_divergence[(y*2)*cur_level_width*2+x*2]
												 + pre_level_divergence[(y*2)*cur_level_width*2+x*2+1]
												 + pre_level_divergence[(y*2+1)*cur_level_width*2+x*2]
												 + pre_level_divergence[(y*2+1)*cur_level_width*2+x*2+1];
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel(float* p_level0, const float* p_level1, const float* p_level2, const float* p_level3, const float* divergence0,
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
			
		if(!leaf0[y*width+x])
			return;
			
		float coeff = 0,sigma = 0;
		float area,len1,len2,weight;
		if(x == 0)
		{
			//left side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf0[y*width+x+1])
			{	
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x+1];
			}
		}
		else if(x == width-1)
		{
			//left side
			if(leaf0[y*width+x-1])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x-1];
			}
			
			//right side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf0[y*width+x-1])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x-1];
			}
			else if(x%2 == 0 && leaf1[(y/2)*(width/2)+x/2-1])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y/2)*(width/2)+x/2-1];
			}
			else if(x%4 == 0 && leaf2[(y/4)*(width/4)+x/4-1])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/4)*(width/4)+x/4-1];
			}
			else if(x%8 == 0 && leaf3[(y/8)*(width/8)+x/8-1])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/8)*(width/8)+x/8-1];
			}

			//right side
			if(leaf0[y*width+x+1])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x+1];
			}
			else if((x+1)%2 == 0 && leaf1[(y/2)*(width/2)+(x+1)/2])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y/2)*(width/2)+(x+1)/2];
			}
			else if((x+1)%4 == 0 && leaf2[(y/4)*(width/4)+(x+1)/4])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/4)*(width/4)+(x+1)/4];
			}
			else if((x+1)%8 == 0 && leaf3[(y/8)*(width/8)+(x+1)/8])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/8)*(width/8)+(x+1)/8];
			}
		}
		
		if(y == 0)
		{
			//left side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf0[(y+1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y+1)*width+x];
			}
		}
		else if(y == height-1)
		{
			//left side
			if(leaf0[(y-1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y-1)*width+x];
			}
			
			//right side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else 
		{
			//left side
			if(leaf0[(y-1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y-1)*width+x];
			}
			else if(y%2 == 0 && leaf1[(y/2-1)*(width/2)+x/2])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y/2-1)*(width/2)+x/2];
			}
			else if(y%4 == 0 && leaf2[(y/4-1)*(width/4)+x/4])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/4-1)*(width/4)+x/4];
			}
			else if(y%8 == 0 && leaf3[(y/8-1)*(width/8)+x/8])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/8-1)*(width/8)+x/8];
			}
			
			//right side
			if(leaf0[(y+1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y+1)*width+x];
			}
			else if((y+1)%2 == 0 && leaf1[(y+1)/2*(width/2)+x/2])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y+1)/2*(width/2)+x/2];
			}
			else if((y+1)%4 == 0 && leaf2[(y+1)/4*(width/4)+x/4])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)/4*(width/4)+x/4];
			}
			else if((y+1)%8 == 0 && leaf3[(y+1)/8*(width/8)+x/8])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)/8*(width/8)+x/8];
			}
		}
		sigma -= divergence0[y*width+x];
		p_level0[y*width+x] = sigma/coeff;	
	}
	
	
	
	__global__
	void SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel(const float* p_level0, float* p_level1, const float* p_level2, const float* p_level3, const float* divergence1,
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		int levelWidth1 = width/2;
		int levelHeight1 = height/2;
		if(x >= levelWidth1 || y >= levelHeight1)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
		
		if(!leaf1[y*levelWidth1+x])
			return;
		
		float coeff = 0.0f, sigma = 0.0f;
		float area,len1,len2,weight;
		if(x == 0)
		{
			//left side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf1[y*levelWidth1+x+1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x+1];
			}
			else //smaller 
			{
				if(leaf0[(y*2)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+(x+1)*2];
				}
				if(leaf0[(y*2+1)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+(x+1)*2];
				}
			} 
		}
		else if(x == levelWidth1 - 1)
		{
			//left side
			if(leaf1[y*levelWidth1+x-1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x-1];
			}
			else //smaller
			{
				if(leaf0[(y*2)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+x*2-1];
				}
				if(leaf0[(y*2+1)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+x*2-1];
				}
			}
			
			//right side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf1[y*levelWidth1+x-1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x-1];
			}
			else if(x%2 == 0 && leaf2[(y/2)*levelWidth1/2+x/2-1])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/2)*levelWidth1/2+x/2-1];
			}
			else if(x%4 == 0 && leaf3[(y/4)*levelWidth1/4+x/4-1])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/4)*levelWidth1/4+x/4-1];
			}
			else //smaller
			{
				if(leaf0[(y*2)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+x*2-1];
				}
				if(leaf0[(y*2+1)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+x*2-1];
				}
			}
			
			//right side
			if(leaf1[y*levelWidth1+x+1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x+1];
			}
			else if((x+1)%2 == 0 && leaf2[(y/2)*levelWidth1/2+(x+1)/2])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/2)*levelWidth1/2+(x+1)/2];
			}
			else if((x+1)%4 == 0 && leaf3[(y/4)*levelWidth1/4+(x+1)/4])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/4)*levelWidth1/4+(x+1)/4];
			}
			else //smaller
			{
				if(leaf0[(y*2)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+(x+1)*2];
				}
				if(leaf0[(y*2+1)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+(x+1)*2];
				}
			}
		}
		
		if(y == 0)
		{
			//left side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf1[(y+1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y+1)*levelWidth1+x];
			}
			else //smaller
			{
				if(leaf0[(y+1)*2*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2];
				}
				if(leaf0[(y+1)*2*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2+1];
				}
			}
		}
		else if(y == levelHeight1 - 1)
		{
			//left side
			if(leaf1[(y-1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y-1)*levelWidth1+x];
			}
			else //smaller
			{
				if(leaf0[(y*2-1)*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2];
				}
				if(leaf0[(y*2-1)*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2+1];
				}
			}
			
			//right side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf1[(y-1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y-1)*levelWidth1+x];
			}
			else if(y%2 == 0 && leaf2[(y/2-1)*levelWidth1/2+x/2])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/2-1)*levelWidth1/2+x/2];
			}
			else if(y%4 == 0 && leaf3[(y/4-1)*levelWidth1/4+x/4])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/4-1)*levelWidth1/4+x/4];
			}
			else //smaller
			{
				if(leaf0[(y*2-1)*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2];
				}
				if(leaf0[(y*2-1)*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2+1];
				}
			}
			
			//right side
			if(leaf1[(y+1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y+1)*levelWidth1+x];
			}
			else if((y+1)%2 == 0 && leaf2[(y+1)/2*levelWidth1/2+x/2])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)/2*levelWidth1/2+x/2];
			}
			else if((y+1)%4 == 0 && leaf3[(y+1)/4*levelWidth1/4+x/4])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)/4*levelWidth1/4+x/4];
			}
			else //smaller
			{
				if(leaf0[(y+1)*2*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2];
				}
				if(leaf0[(y+1)*2*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2+1];
				}
			}
		}
		sigma -= divergence1[y*levelWidth1+x];
		p_level1[y*levelWidth1+x] = sigma/coeff;
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel(const float* p_level0, const float* p_level1, float* p_level2, const float* p_level3, const float* divergence2,
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		int levelWidth1 = width/2;
		//int levelHeight1 = height/2;
		int levelWidth2 = width/4;
		int levelHeight2 = height/4;
		if(x >= levelWidth2 || y >= levelHeight2)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
		
		if(!leaf2[y*levelWidth2+x])
			return;
		
		float coeff = 0.0f, sigma = 0.0f;
		float area,len1,len2,weight;
		if(x == 0)
		{
			//left side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf2[y*levelWidth2+x+1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x+1];
			} 
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+(x+1)*4];
					}
					if(leaf0[(y*4+1)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+(x+1)*4];
					}
				}
				
				if(leaf1[(y*2+1)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+(x+1)*4];
					}
					if(leaf0[(y*4+3)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+(x+1)*4];
					}
				}
			}
			
		}
		else if(x == levelWidth2 - 1)
		{
			// left side
			if(leaf2[y*levelWidth2+x-1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x-1];
			}
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+x*4-1];
					}
					if(leaf0[(y*4+1)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+x*4-1];
					}
				}
				if(leaf1[(y*2+1)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+x*4-1];
					}
					if(leaf0[(y*4+3)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+x*4-1];
					}
				}
			}
			
			//right side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			// left side
			if(leaf2[y*levelWidth2+x-1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x-1];
			}
			else if(x%2 == 0 && leaf3[(y/2)*levelWidth2/2+x/2-1])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/2)*levelWidth2/2+x/2-1];
			}
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+x*4-1];
					}
					if(leaf0[(y*4+1)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+x*4-1];
					}
				}
				
				if(leaf1[(y*2+1)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+x*4-1];
					}
					if(leaf0[(y*4+3)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+x*4-1];
					}
				}
			}
			
			//right side
			if(leaf2[y*levelWidth2+x+1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x+1];
			}
			else if((x+1)%2 == 0 && leaf3[(y/2)*levelWidth2/2+(x+1)/2])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/2)*levelWidth2/2+(x+1)/2];
			}
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+(x+1)*4];
					}
					if(leaf0[(y*4+1)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+(x+1)*4];
					}
				}
				
				if(leaf1[(y*2+1)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+(x+1)*4];
					}
					if(leaf0[(y*4+3)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+(x+1)*4];
					}
				}
			}
		}
		
		if(y == 0)
		{
			//left side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf2[(y+1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)*levelWidth2+x];
			}
			else //smaller
			{
				if(leaf1[(y+1)*2*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4];
					}
					if(leaf0[(y+1)*4*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+1];
					}
				}
				
				if(leaf1[(y+1)*2*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+2];
					}
					if(leaf0[(y+1)*4*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+3];
					}
				}
			}
		}
		else if(y == levelHeight2 - 1)
		{
			//left side
			if(leaf2[(y-1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y-1)*levelWidth2+x];
			}
			else //smaller
			{
				if(leaf1[(y*2-1)*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4];
					}
					if(leaf0[(y*4-1)*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+1];
					}
				}
				
				if(leaf1[(y*2-1)*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+2];
					}
					if(leaf0[(y*4-1)*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+3];
					}
				}
			}
			
			//right side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf2[(y-1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y-1)*levelWidth2+x];
			}
			else if(y%2 == 0 && leaf3[(y/2-1)*levelWidth2/2+x/2])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/2-1)*levelWidth2/2+x/2];
			}
			else //smaller
			{
				if(leaf1[(y*2-1)*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4];
					}
					if(leaf0[(y*4-1)*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+1];
					}
				}
				
				if(leaf1[(y*2-1)*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+2];
					}
					if(leaf0[(y*4-1)*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+3];
					}
				}
			}
			
			//right side
			if(leaf2[(y+1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)*levelWidth2+x];
			}
			else if((y+1)%2 == 0 && leaf3[(y+1)/2*levelWidth2/2+x/2])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)/2*levelWidth2/2+x/2];
			}
			else //smaller
			{
				if(leaf1[(y+1)*2*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4];
					}
					if(leaf0[(y+1)*4*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+1];
					}
				}
				
				if(leaf1[(y+1)*2*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+2];
					}
					if(leaf0[(y+1)*4*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+3];
					}
				}
			}
		}
		
		sigma -= divergence2[y*levelWidth2+x];
		p_level2[y*levelWidth2+x] = sigma/coeff;
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel(const float* p_level0, const float* p_level1, const float* p_level2, float* p_level3, const float* divergence3,
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, const bool redkernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		int levelWidth1 = width/2;
		//int levelHeight1 = height/2;
		int levelWidth2 = width/4;
		//int levelHeight2 = height/4;
		int levelWidth3 = width/8;
		int levelHeight3 = height/8;
		if(x >= levelWidth3 || y >= levelHeight3)
			return ;

		int rest = (x+y)%2;

		if(rest == (redkernel ? 1 : 0))
			return;
		
		if(!leaf3[y*levelWidth3+x])
			return;
		
		float coeff = 0.0f, sigma = 0.0f;
		float area,len1,len2,weight;
		
		//X left side
		if(x == 0)
		{
			//left side
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			if(leaf3[y*levelWidth3+x-1])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[y*levelWidth3+x-1];
			}
			else //smaller
			{
				if(leaf2[(y*2)*levelWidth2+x*2-1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2)*levelWidth2+x*2-1];
				}
				else //smaller
				{
					if(leaf1[(y*4)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8)*width+x*8-1];
						}
						if(leaf0[(y*8+1)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+1)*width+x*8-1];
						}
					}
					
					if(leaf1[(y*4+1)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+1)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8+2)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+2)*width+x*8-1];
						}
						if(leaf0[(y*8+3)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*+3)*width+x*8-1];
						}
					}
				}
				
				if(leaf2[(y*2+1)*levelWidth2+x*2-1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2+1)*levelWidth2+x*2-1];
				}
				else //smaller
				{
					if(leaf1[(y*4+2)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+2)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8+4)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+4)*width+x*8-1];
						}
						if(leaf0[(y*8+5)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+5)*width+x*8-1];
						}
					}
					
					if(leaf1[(y*4+3)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+3)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8+6)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+6)*width+x*8-1];
						}
						if(leaf0[(y*8+7)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+7)*width+x*7-1];
						}
					}
				}
			}
		}
		
		//X right side
		if(x == levelWidth3-1)
		{
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			if(leaf3[y*levelWidth3+x+1])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[y*levelWidth3+x+1];
			}
			else //smaller
			{
				if(leaf2[(y*2)*levelWidth2+(x+1)*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2)*levelWidth2+(x+1)*2];
				}
				else //smaller
				{
					if(leaf1[(y*4)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8)*width+(x+1)*8];
						}
						if(leaf0[(y*8+1)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+1)*width+(x+1)*8];
						}
					}
					
					if(leaf1[(y*4+1)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+1)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8+2)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+2)*width+(x+1)*8];
						}
						if(leaf0[(y*8+3)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+3)*width+(x+1)*8];
						}
					}
				}
				
				if(leaf2[(y*2+1)*levelWidth2+(x+1)*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2+1)*levelWidth2+(x+1)*2];
				}
				else //smaller
				{
					if(leaf1[(y*4+2)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+2)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8+4)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+4)*width+(x+1)*8];
						}
						if(leaf0[(y*8+5)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+5)*width+(x+1)*8];
						}
					}
					
					if(leaf1[(y*4+3)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+3)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8+6)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+6)*width+(x+1)*8];
						}
						if(leaf0[(y*8+7)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+7)*width+(x+1)*8];
						}
					}
				}
			}
		}
		
		//Y left side
		if(y == 0)
		{
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;	
		}
		else
		{
			if(leaf3[(y-1)*levelWidth3+x])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y-1)*levelWidth3+x];
			}
			else //smaller
			{
				if(leaf2[(y*2-1)*levelWidth2+x*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2-1)*levelWidth2+x*2];
				}
				else //smaller
				{
					if(leaf1[(y*4-1)*levelWidth1+x*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4-1)*levelWidth1+x*4];
					}
					else //smaller
					{
						if(leaf0[(y*8-1)*width+x*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8];
						}
						if(leaf0[(y*8-1)*width+x*8+1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+1];
						}
					}
				}
				
				if(leaf2[(y*2-1)*levelWidth2+x*2+1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2-1)*levelWidth2+x*2+1];
				}
				else //smaller
				{
					if(leaf1[(y*4-1)*levelWidth1+x*4+2])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4-1)*levelWidth1+x*4+2];
					}
					else //smaller
					{
						if(leaf0[(y*8-1)*width+x*8+4])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+4];
						}
						if(leaf0[(y*8-1)*width+x*8+5])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+5];
						}
					}
					
					if(leaf1[(y*4-1)*levelWidth1+x*4+3])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4-1)*levelWidth1+x*4+3];
					}
					else //smaller
					{
						if(leaf0[(y*8-1)*width+x*8+6])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+6];
						}
						if(leaf0[(y*8-1)*width+x*8+7])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+7];
						}
					}
				}
			}
		}
		
		//Y right side
		if(y == levelHeight3 - 1)
		{
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			if(leaf3[(y+1)*levelWidth3+x])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)*levelWidth3+x];
			}
			else //smaller
			{
				if(leaf2[(y+1)*2*levelWidth2+x*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y+1)*2*levelWidth2+x*2];
				}
				else //smaller
				{
					if(leaf1[(y+1)*4*levelWidth1+x*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4];
					} 
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8];
						}
						if(leaf0[(y+1)*8*width+x*8+1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+1];
						}
					}
					
					if(leaf1[(y+1)*4*levelWidth1+x*4+1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4+1];
					}
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8+2])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+2];
						}
						if(leaf0[(y+1)*8*width+x*8+3])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+3];
						}
					}
				}
				
				if(leaf2[(y+1)*2*levelWidth2+x*2+1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y+1)*2*levelWidth2+x*2+1];
				}
				else //smaller
				{
					if(leaf1[(y+1)*4*levelWidth1+x*4+2])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4+2];
					}
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8+4])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+4];
						}
						if(leaf0[(y+1)*8*width+x*8+5])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+5];
						}
					}
					
					if(leaf1[(y+1)*4*levelWidth1+x*4+3])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4+3];
					}
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8+6])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+6];
						}
						if(leaf0[(y+1)*8*width+x*8+7])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+7];
						}
					}
				}	
			}
		}
		
		sigma -= divergence3[y*levelWidth3+x];
		p_level3[y*levelWidth3+x] = sigma/coeff;
	}
	
	/*** Another Implementation of Open Octree Poisson ***/											
	__global__
	void SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel(float* p_level0, const float* p_level1, const float* p_level2, const float* p_level3, const float* divergence0, 
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, 
														const int level0_num, const int* level0_index)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level0_num)
			return ;
			
		int x = level0_index[cur_idx*2+0];
		int y = level0_index[cur_idx*2+1];
		
		float coeff = 0,sigma = 0;
		float area,len1,len2,weight;
		if(x == 0)
		{
			//left side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf0[y*width+x+1])
			{	
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x+1];
			}
		}
		else if(x == width-1)
		{
			//left side
			if(leaf0[y*width+x-1])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x-1];
			}
			
			//right side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf0[y*width+x-1])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x-1];
			}
			else if(x%2 == 0 && leaf1[(y/2)*(width/2)+x/2-1])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y/2)*(width/2)+x/2-1];
			}
			else if(x%4 == 0 && leaf2[(y/4)*(width/4)+x/4-1])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/4)*(width/4)+x/4-1];
			}
			else if(x%8 == 0 && leaf3[(y/8)*(width/8)+x/8-1])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/8)*(width/8)+x/8-1];
			}

			//right side
			if(leaf0[y*width+x+1])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[y*width+x+1];
			}
			else if((x+1)%2 == 0 && leaf1[(y/2)*(width/2)+(x+1)/2])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y/2)*(width/2)+(x+1)/2];
			}
			else if((x+1)%4 == 0 && leaf2[(y/4)*(width/4)+(x+1)/4])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/4)*(width/4)+(x+1)/4];
			}
			else if((x+1)%8 == 0 && leaf3[(y/8)*(width/8)+(x+1)/8])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/8)*(width/8)+(x+1)/8];
			}
		}
		
		if(y == 0)
		{
			//left side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf0[(y+1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y+1)*width+x];
			}
		}
		else if(y == height-1)
		{
			//left side
			if(leaf0[(y-1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y-1)*width+x];
			}
			
			//right side
			area = 1.0f;
			len1 = 4.0f;
			len2 = 0.5f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else 
		{
			//left side
			if(leaf0[(y-1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y-1)*width+x];
			}
			else if(y%2 == 0 && leaf1[(y/2-1)*(width/2)+x/2])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y/2-1)*(width/2)+x/2];
			}
			else if(y%4 == 0 && leaf2[(y/4-1)*(width/4)+x/4])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/4-1)*(width/4)+x/4];
			}
			else if(y%8 == 0 && leaf3[(y/8-1)*(width/8)+x/8])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/8-1)*(width/8)+x/8];
			}
			
			//right side
			if(leaf0[(y+1)*width+x])
			{
				area = 1.0f;
				len1 = 0.5f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level0[(y+1)*width+x];
			}
			else if((y+1)%2 == 0 && leaf1[(y+1)/2*(width/2)+x/2])
			{
				area = 1.0f;
				len1 = 1.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y+1)/2*(width/2)+x/2];
			}
			else if((y+1)%4 == 0 && leaf2[(y+1)/4*(width/4)+x/4])
			{
				area = 1.0f;
				len1 = 2.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)/4*(width/4)+x/4];
			}
			else if((y+1)%8 == 0 && leaf3[(y+1)/8*(width/8)+x/8])
			{
				area = 1.0f;
				len1 = 4.0f;
				len2 = 0.5f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)/8*(width/8)+x/8];
			}
		}
		sigma -= divergence0[y*width+x];
		p_level0[y*width+x] = sigma/coeff;	
		
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel(const float* p_level0, float* p_level1, const float* p_level2, const float* p_level3, const float* divergence1, 
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, 
														const int level1_num, const int* level1_index)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level1_num)
			return ;
			
		int x = level1_index[cur_idx*2+0];
		int y = level1_index[cur_idx*2+1];
		
		int levelWidth1 = width/2;
		int levelHeight1 = height/2;	
		float coeff = 0.0f, sigma = 0.0f;
		float area,len1,len2,weight;
		if(x == 0)
		{
			//left side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf1[y*levelWidth1+x+1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x+1];
			}
			else //smaller 
			{
				if(leaf0[(y*2)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+(x+1)*2];
				}
				if(leaf0[(y*2+1)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+(x+1)*2];
				}
			} 
		}
		else if(x == levelWidth1 - 1)
		{
			//left side
			if(leaf1[y*levelWidth1+x-1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x-1];
			}
			else //smaller
			{
				if(leaf0[(y*2)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+x*2-1];
				}
				if(leaf0[(y*2+1)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+x*2-1];
				}
			}
			
			//right side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf1[y*levelWidth1+x-1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x-1];
			}
			else if(x%2 == 0 && leaf2[(y/2)*levelWidth1/2+x/2-1])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/2)*levelWidth1/2+x/2-1];
			}
			else if(x%4 == 0 && leaf3[(y/4)*levelWidth1/4+x/4-1])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/4)*levelWidth1/4+x/4-1];
			}
			else //smaller
			{
				if(leaf0[(y*2)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+x*2-1];
				}
				if(leaf0[(y*2+1)*width+x*2-1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+x*2-1];
				}
			}
			
			//right side
			if(leaf1[y*levelWidth1+x+1])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[y*levelWidth1+x+1];
			}
			else if((x+1)%2 == 0 && leaf2[(y/2)*levelWidth1/2+(x+1)/2])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/2)*levelWidth1/2+(x+1)/2];
			}
			else if((x+1)%4 == 0 && leaf3[(y/4)*levelWidth1/4+(x+1)/4])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/4)*levelWidth1/4+(x+1)/4];
			}
			else //smaller
			{
				if(leaf0[(y*2)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2)*width+(x+1)*2];
				}
				if(leaf0[(y*2+1)*width+(x+1)*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2+1)*width+(x+1)*2];
				}
			}
		}
		
		if(y == 0)
		{
			//left side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf1[(y+1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y+1)*levelWidth1+x];
			}
			else //smaller
			{
				if(leaf0[(y+1)*2*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2];
				}
				if(leaf0[(y+1)*2*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2+1];
				}
			}
		}
		else if(y == levelHeight1 - 1)
		{
			//left side
			if(leaf1[(y-1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y-1)*levelWidth1+x];
			}
			else //smaller
			{
				if(leaf0[(y*2-1)*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2];
				}
				if(leaf0[(y*2-1)*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2+1];
				}
			}
			
			//right side
			area = 2.0f;
			len1 = 4.0f;
			len2 = 1.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf1[(y-1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y-1)*levelWidth1+x];
			}
			else if(y%2 == 0 && leaf2[(y/2-1)*levelWidth1/2+x/2])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y/2-1)*levelWidth1/2+x/2];
			}
			else if(y%4 == 0 && leaf3[(y/4-1)*levelWidth1/4+x/4])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/4-1)*levelWidth1/4+x/4];
			}
			else //smaller
			{
				if(leaf0[(y*2-1)*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2];
				}
				if(leaf0[(y*2-1)*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y*2-1)*width+x*2+1];
				}
			}
			
			//right side
			if(leaf1[(y+1)*levelWidth1+x])
			{
				area = 2.0f;
				len1 = 1.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level1[(y+1)*levelWidth1+x];
			}
			else if((y+1)%2 == 0 && leaf2[(y+1)/2*levelWidth1/2+x/2])
			{
				area = 2.0f;
				len1 = 2.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)/2*levelWidth1/2+x/2];
			}
			else if((y+1)%4 == 0 && leaf3[(y+1)/4*levelWidth1/4+x/4])
			{
				area = 2.0f;
				len1 = 4.0f;
				len2 = 1.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)/4*levelWidth1/4+x/4];
			}
			else //smaller
			{
				if(leaf0[(y+1)*2*width+x*2])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2];
				}
				if(leaf0[(y+1)*2*width+x*2+1])
				{
					area = 1.0f;
					len1 = 1.0f;
					len2 = 0.5f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level0[(y+1)*2*width+x*2+1];
				}
			}
		}
		sigma -= divergence1[y*levelWidth1+x];
		p_level1[y*levelWidth1+x] = sigma/coeff;	
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel(const float* p_level0, const float* p_level1, float* p_level2, const float* p_level3, const float* divergence2, 
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, 
														const int level2_num, const int* level2_index)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level2_num)
			return ;
		
		int x = level2_index[cur_idx*2+0];
		int y = level2_index[cur_idx*2+1];
		
		int levelWidth1 = width/2;
		//int levelHeight1 = height/2;
		int levelWidth2 = width/4;
		int levelHeight2 = height/4;
		float coeff = 0.0f, sigma = 0.0f;
		float area,len1,len2,weight;
		if(x == 0)
		{
			//left side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf2[y*levelWidth2+x+1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x+1];
			} 
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+(x+1)*4];
					}
					if(leaf0[(y*4+1)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+(x+1)*4];
					}
				}
				
				if(leaf1[(y*2+1)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+(x+1)*4];
					}
					if(leaf0[(y*4+3)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+(x+1)*4];
					}
				}
			}
			
		}
		else if(x == levelWidth2 - 1)
		{
			// left side
			if(leaf2[y*levelWidth2+x-1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x-1];
			}
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+x*4-1];
					}
					if(leaf0[(y*4+1)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+x*4-1];
					}
				}
				if(leaf1[(y*2+1)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+x*4-1];
					}
					if(leaf0[(y*4+3)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+x*4-1];
					}
				}
			}
			
			//right side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			// left side
			if(leaf2[y*levelWidth2+x-1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x-1];
			}
			else if(x%2 == 0 && leaf3[(y/2)*levelWidth2/2+x/2-1])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/2)*levelWidth2/2+x/2-1];
			}
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+x*4-1];
					}
					if(leaf0[(y*4+1)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+x*4-1];
					}
				}
				
				if(leaf1[(y*2+1)*levelWidth1+x*2-1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+x*2-1];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+x*4-1];
					}
					if(leaf0[(y*4+3)*width+x*4-1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+x*4-1];
					}
				}
			}
			
			//right side
			if(leaf2[y*levelWidth2+x+1])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[y*levelWidth2+x+1];
			}
			else if((x+1)%2 == 0 && leaf3[(y/2)*levelWidth2/2+(x+1)/2])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/2)*levelWidth2/2+(x+1)/2];
			}
			else //smaller
			{
				if(leaf1[(y*2)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4)*width+(x+1)*4];
					}
					if(leaf0[(y*4+1)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+1)*width+(x+1)*4];
					}
				}
				
				if(leaf1[(y*2+1)*levelWidth1+(x+1)*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2+1)*levelWidth1+(x+1)*2];
				}
				else //smaller
				{
					if(leaf0[(y*4+2)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+2)*width+(x+1)*4];
					}
					if(leaf0[(y*4+3)*width+(x+1)*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4+3)*width+(x+1)*4];
					}
				}
			}
		}
		
		if(y == 0)
		{
			//left side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
			
			//right side
			if(leaf2[(y+1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)*levelWidth2+x];
			}
			else //smaller
			{
				if(leaf1[(y+1)*2*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4];
					}
					if(leaf0[(y+1)*4*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+1];
					}
				}
				
				if(leaf1[(y+1)*2*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+2];
					}
					if(leaf0[(y+1)*4*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+3];
					}
				}
			}
		}
		else if(y == levelHeight2 - 1)
		{
			//left side
			if(leaf2[(y-1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y-1)*levelWidth2+x];
			}
			else //smaller
			{
				if(leaf1[(y*2-1)*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4];
					}
					if(leaf0[(y*4-1)*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+1];
					}
				}
				
				if(leaf1[(y*2-1)*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+2];
					}
					if(leaf0[(y*4-1)*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+3];
					}
				}
			}
			
			//right side
			area = 4.0f;
			len1 = 4.0f;
			len2 = 2.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			//left side
			if(leaf2[(y-1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y-1)*levelWidth2+x];
			}
			else if(y%2 == 0 && leaf3[(y/2-1)*levelWidth2/2+x/2])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y/2-1)*levelWidth2/2+x/2];
			}
			else //smaller
			{
				if(leaf1[(y*2-1)*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4];
					}
					if(leaf0[(y*4-1)*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+1];
					}
				}
				
				if(leaf1[(y*2-1)*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y*2-1)*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y*4-1)*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+2];
					}
					if(leaf0[(y*4-1)*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y*4-1)*width+x*4+3];
					}
				}
			}
			
			//right side
			if(leaf2[(y+1)*levelWidth2+x])
			{
				area = 4.0f;
				len1 = 2.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level2[(y+1)*levelWidth2+x];
			}
			else if((y+1)%2 == 0 && leaf3[(y+1)/2*levelWidth2/2+x/2])
			{
				area = 4.0f;
				len1 = 4.0f;
				len2 = 2.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)/2*levelWidth2/2+x/2];
			}
			else //smaller
			{
				if(leaf1[(y+1)*2*levelWidth1+x*2])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4];
					}
					if(leaf0[(y+1)*4*width+x*4+1])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+1];
					}
				}
				
				if(leaf1[(y+1)*2*levelWidth1+x*2+1])
				{
					area = 2.0f;
					len1 = 2.0f;
					len2 = 1.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level1[(y+1)*2*levelWidth1+x*2+1];
				}
				else //smaller
				{
					if(leaf0[(y+1)*4*width+x*4+2])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+2];
					}
					if(leaf0[(y+1)*4*width+x*4+3])
					{
						area = 1.0f;
						len1 = 2.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level0[(y+1)*4*width+x*4+3];
					}
				}
			}
		}
		
		sigma -= divergence2[y*levelWidth2+x];
		p_level2[y*levelWidth2+x] = sigma/coeff;
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel(const float* p_level0, const float* p_level1, const float* p_level2, float* p_level3, const float* divergence3, 
														const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, const int width, const int height, 
														const int level3_num, const int* level3_index)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;

		if(cur_idx >= level3_num)
			return ;

		int x = level3_index[cur_idx*2+0];
		int y = level3_index[cur_idx*2+1];
		int levelWidth1 = width/2;
		//int levelHeight1 = height/2;
		int levelWidth2 = width/4;
		//int levelHeight2 = height/4;
		int levelWidth3 = width/8;
		int levelHeight3 = height/8;
		float coeff = 0.0f, sigma = 0.0f;
		float area,len1,len2,weight;
		
		//X left side
		if(x == 0)
		{
			//left side
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			if(leaf3[y*levelWidth3+x-1])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[y*levelWidth3+x-1];
			}
			else //smaller
			{
				if(leaf2[(y*2)*levelWidth2+x*2-1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2)*levelWidth2+x*2-1];
				}
				else //smaller
				{
					if(leaf1[(y*4)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8)*width+x*8-1];
						}
						if(leaf0[(y*8+1)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+1)*width+x*8-1];
						}
					}
					
					if(leaf1[(y*4+1)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+1)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8+2)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+2)*width+x*8-1];
						}
						if(leaf0[(y*8+3)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*+3)*width+x*8-1];
						}
					}
				}
				
				if(leaf2[(y*2+1)*levelWidth2+x*2-1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2+1)*levelWidth2+x*2-1];
				}
				else //smaller
				{
					if(leaf1[(y*4+2)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+2)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8+4)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+4)*width+x*8-1];
						}
						if(leaf0[(y*8+5)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+5)*width+x*8-1];
						}
					}
					
					if(leaf1[(y*4+3)*levelWidth1+x*4-1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 0.5f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+3)*levelWidth1+x*4-1];
					}
					else //smaller
					{
						if(leaf0[(y*8+6)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+6)*width+x*8-1];
						}
						if(leaf0[(y*8+7)*width+x*8-1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+7)*width+x*7-1];
						}
					}
				}
			}
		}
		
		//X right side
		if(x == levelWidth3-1)
		{
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			if(leaf3[y*levelWidth3+x+1])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[y*levelWidth3+x+1];
			}
			else //smaller
			{
				if(leaf2[(y*2)*levelWidth2+(x+1)*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2)*levelWidth2+(x+1)*2];
				}
				else //smaller
				{
					if(leaf1[(y*4)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8)*width+(x+1)*8];
						}
						if(leaf0[(y*8+1)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+1)*width+(x+1)*8];
						}
					}
					
					if(leaf1[(y*4+1)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+1)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8+2)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+2)*width+(x+1)*8];
						}
						if(leaf0[(y*8+3)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+3)*width+(x+1)*8];
						}
					}
				}
				
				if(leaf2[(y*2+1)*levelWidth2+(x+1)*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2+1)*levelWidth2+(x+1)*2];
				}
				else //smaller
				{
					if(leaf1[(y*4+2)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+2)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8+4)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+4)*width+(x+1)*8];
						}
						if(leaf0[(y*8+5)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+5)*width+(x+1)*8];
						}
					}
					
					if(leaf1[(y*4+3)*levelWidth1+(x+1)*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4+3)*levelWidth1+(x+1)*4];
					}
					else //smaller
					{
						if(leaf0[(y*8+6)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+6)*width+(x+1)*8];
						}
						if(leaf0[(y*8+7)*width+(x+1)*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8+7)*width+(x+1)*8];
						}
					}
				}
			}
		}
		
		//Y left side
		if(y == 0)
		{
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;	
		}
		else
		{
			if(leaf3[(y-1)*levelWidth3+x])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y-1)*levelWidth3+x];
			}
			else //smaller
			{
				if(leaf2[(y*2-1)*levelWidth2+x*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2-1)*levelWidth2+x*2];
				}
				else //smaller
				{
					if(leaf1[(y*4-1)*levelWidth1+x*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4-1)*levelWidth1+x*4];
					}
					else //smaller
					{
						if(leaf0[(y*8-1)*width+x*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8];
						}
						if(leaf0[(y*8-1)*width+x*8+1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+1];
						}
					}
				}
				
				if(leaf2[(y*2-1)*levelWidth2+x*2+1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y*2-1)*levelWidth2+x*2+1];
				}
				else //smaller
				{
					if(leaf1[(y*4-1)*levelWidth1+x*4+2])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4-1)*levelWidth1+x*4+2];
					}
					else //smaller
					{
						if(leaf0[(y*8-1)*width+x*8+4])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+4];
						}
						if(leaf0[(y*8-1)*width+x*8+5])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+5];
						}
					}
					
					if(leaf1[(y*4-1)*levelWidth1+x*4+3])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y*4-1)*levelWidth1+x*4+3];
					}
					else //smaller
					{
						if(leaf0[(y*8-1)*width+x*8+6])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+6];
						}
						if(leaf0[(y*8-1)*width+x*8+7])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y*8-1)*width+x*8+7];
						}
					}
				}
			}
		}
		
		//Y right side
		if(y == levelHeight3 - 1)
		{
			area = 8.0f;
			len1 = 4.0f;
			len2 = 4.0f;
			weight = area/(len1+len2);
			coeff += weight;
		}
		else
		{
			if(leaf3[(y+1)*levelWidth3+x])
			{
				area = 8.0f;
				len1 = 4.0f;
				len2 = 4.0f;
				weight = area/(len1+len2);
				coeff += weight;
				sigma += weight*p_level3[(y+1)*levelWidth3+x];
			}
			else //smaller
			{
				if(leaf2[(y+1)*2*levelWidth2+x*2])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y+1)*2*levelWidth2+x*2];
				}
				else //smaller
				{
					if(leaf1[(y+1)*4*levelWidth1+x*4])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4];
					} 
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8];
						}
						if(leaf0[(y+1)*8*width+x*8+1])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+1];
						}
					}
					
					if(leaf1[(y+1)*4*levelWidth1+x*4+1])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4+1];
					}
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8+2])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+2];
						}
						if(leaf0[(y+1)*8*width+x*8+3])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+3];
						}
					}
				}
				
				if(leaf2[(y+1)*2*levelWidth2+x*2+1])
				{
					area = 4.0f;
					len1 = 4.0f;
					len2 = 2.0f;
					weight = area/(len1+len2);
					coeff += weight;
					sigma += weight*p_level2[(y+1)*2*levelWidth2+x*2+1];
				}
				else //smaller
				{
					if(leaf1[(y+1)*4*levelWidth1+x*4+2])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4+2];
					}
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8+4])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+4];
						}
						if(leaf0[(y+1)*8*width+x*8+5])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+5];
						}
					}
					
					if(leaf1[(y+1)*4*levelWidth1+x*4+3])
					{
						area = 2.0f;
						len1 = 4.0f;
						len2 = 1.0f;
						weight = area/(len1+len2);
						coeff += weight;
						sigma += weight*p_level1[(y+1)*4*levelWidth1+x*4+3];
					}
					else //smaller
					{
						if(leaf0[(y+1)*8*width+x*8+6])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+6];
						}
						if(leaf0[(y+1)*8*width+x*8+7])
						{
							area = 1.0f;
							len1 = 4.0f;
							len2 = 0.5f;
							weight = area/(len1+len2);
							coeff += weight;
							sigma += weight*p_level0[(y+1)*8*width+x*8+7];
						}
					}
				}	
			}
		}
		
		sigma -= divergence3[y*levelWidth3+x];
		p_level3[y*levelWidth3+x] = sigma/coeff;
	}
	
	/*** Third Implementation of Open Octree Poisson ***/											
	__global__
	void SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel(float* p_level0, const float* p_level1, const float* p_level2, const float* p_level3, const float* divergence0, 
														const int width, const int height, 
														const int level0_num, const int* level0_index, const int* level0_neighborinfo)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level0_num)
			return ;
		
		const int index_channels = 4;
		const int neighborinfo_channels = 3;
		
		int x = level0_index[cur_idx*index_channels+0];
		int y = level0_index[cur_idx*index_channels+1];
		int num = level0_index[cur_idx*index_channels+2];
		int offset = level0_index[cur_idx*index_channels+3];
		
		float coeff = 0,sigma = 0;
		const float weight[4] = {1.0f/(0.5f+0.5f),1.0f/(0.5f+1.0f),1.0f/(0.5f+2.0f),1.0f/(0.5f+4.0f)};
		int imWidth[4] = {width,width/2,width/4,width/8};
		int imHeight[4] = {height,height/2,height/4,height/8};
		const float* pressure[4] = {p_level0,p_level1,p_level2,p_level3};
		for(int nn = 0;nn < num;nn++)
		{
			int neighbor_offset = (offset+nn)*neighborinfo_channels;
			int neighbor_lvl = level0_neighborinfo[neighbor_offset+0];
			int neighbor_x = level0_neighborinfo[neighbor_offset+1]; 
			int neighbor_y = level0_neighborinfo[neighbor_offset+2];
			
			float cur_weight = weight[neighbor_lvl];
			coeff += cur_weight;
			if(neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < imWidth[neighbor_lvl] && neighbor_y < imHeight[neighbor_lvl])
				sigma += cur_weight * pressure[neighbor_lvl][neighbor_y*imWidth[neighbor_lvl]+neighbor_x];
		}
		
		sigma -= divergence0[y*width+x];
		p_level0[y*width+x] = sigma/coeff;		
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel(const float* p_level0, float* p_level1, const float* p_level2, const float* p_level3, const float* divergence1, 
														const int width, const int height, 
														const int level1_num, const int* level1_index, const int* level1_neighborinfo)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level1_num)
			return ;
		
		const int index_channels = 4;
		const int neighborinfo_channels = 3;
		
		int x = level1_index[cur_idx*index_channels+0];
		int y = level1_index[cur_idx*index_channels+1];
		int num = level1_index[cur_idx*index_channels+2];
		int offset = level1_index[cur_idx*index_channels+3];
		
		float coeff = 0,sigma = 0;
		const float weight[4] = {1.0f/(1.0f+0.5f),2.0f/(1.0f+1.0f),2.0f/(1.0f+2.0f),2.0f/(1.0f+4.0f)};
		int imWidth[4] = {width,width/2,width/4,width/8};
		int imHeight[4] = {height,height/2,height/4,height/8};
		const float* pressure[4] = {p_level0,p_level1,p_level2,p_level3};
		for(int nn = 0;nn < num;nn++)
		{
			int neighbor_offset = (offset+nn)*neighborinfo_channels;
			int neighbor_lvl = level1_neighborinfo[neighbor_offset+0];
			int neighbor_x = level1_neighborinfo[neighbor_offset+1]; 
			int neighbor_y = level1_neighborinfo[neighbor_offset+2];
			
			float cur_weight = weight[neighbor_lvl];
			coeff += cur_weight;
			if(neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < imWidth[neighbor_lvl] && neighbor_y < imHeight[neighbor_lvl])
				sigma += cur_weight * pressure[neighbor_lvl][neighbor_y*imWidth[neighbor_lvl]+neighbor_x];
		}
		
		sigma -= divergence1[y*width/2+x];
		p_level1[y*width/2+x] = sigma/coeff;		
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel(const float* p_level0, const float* p_level1, float* p_level2, const float* p_level3, const float* divergence2, 
														const int width, const int height, 
														const int level2_num, const int* level2_index, const int* level2_neighborinfo)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level2_num)
			return ;
		
		const int index_channels = 4;
		const int neighborinfo_channels = 3;
		
		int x = level2_index[cur_idx*index_channels+0];
		int y = level2_index[cur_idx*index_channels+1];
		int num = level2_index[cur_idx*index_channels+2];
		int offset = level2_index[cur_idx*index_channels+3];
		
		float coeff = 0,sigma = 0;
		const float weight[4] = {1.0f/(2.0f+0.5f),2.0f/(2.0f+1.0f),4.0f/(2.0f+2.0f),4.0f/(2.0f+4.0f)};
		int imWidth[4] = {width,width/2,width/4,width/8};
		int imHeight[4] = {height,height/2,height/4,height/8};
		const float* pressure[4] = {p_level0,p_level1,p_level2,p_level3};
		for(int nn = 0;nn < num;nn++)
		{
			int neighbor_offset = (offset+nn)*neighborinfo_channels;
			int neighbor_lvl = level2_neighborinfo[neighbor_offset+0];
			int neighbor_x = level2_neighborinfo[neighbor_offset+1]; 
			int neighbor_y = level2_neighborinfo[neighbor_offset+2];
			
			float cur_weight = weight[neighbor_lvl];
			coeff += cur_weight;
			if(neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < imWidth[neighbor_lvl] && neighbor_y < imHeight[neighbor_lvl])
				sigma += cur_weight * pressure[neighbor_lvl][neighbor_y*imWidth[neighbor_lvl]+neighbor_x];
		}
		
		sigma -= divergence2[y*width/4+x];
		p_level2[y*width/4+x] = sigma/coeff;		
	}
	
	__global__
	void SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel(const float* p_level0, const float* p_level1, const float* p_level2, float* p_level3, const float* divergence3, 
														const int width, const int height, 
														const int level3_num, const int* level3_index, const int* level3_neighborinfo)
	{
		int cur_idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if( cur_idx >= level3_num)
			return ;
		
		const int index_channels = 4;
		const int neighborinfo_channels = 3;
		
		int x = level3_index[cur_idx*index_channels+0];
		int y = level3_index[cur_idx*index_channels+1];
		int num = level3_index[cur_idx*index_channels+2];
		int offset = level3_index[cur_idx*index_channels+3];
		
		float coeff = 0,sigma = 0;
		const float weight[4] = {1.0f/(4.0f+0.5f),2.0f/(4.0f+1.0f),4.0f/(4.0f+2.0f),8.0f/(4.0f+4.0f)};
		int imWidth[4] = {width,width/2,width/4,width/8};
		int imHeight[4] = {height,height/2,height/4,height/8};
		const float* pressure[4] = {p_level0,p_level1,p_level2,p_level3};
		for(int nn = 0;nn < num;nn++)
		{
			int neighbor_offset = (offset+nn)*neighborinfo_channels;
			int neighbor_lvl = level3_neighborinfo[neighbor_offset+0];
			int neighbor_x = level3_neighborinfo[neighbor_offset+1]; 
			int neighbor_y = level3_neighborinfo[neighbor_offset+2];
			
			float cur_weight = weight[neighbor_lvl];
			coeff += cur_weight;
			if(neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < imWidth[neighbor_lvl] && neighbor_y < imHeight[neighbor_lvl])
				sigma += cur_weight * pressure[neighbor_lvl][neighbor_y*imWidth[neighbor_lvl]+neighbor_x];
		}
		
		sigma -= divergence3[y*width/8+x];
		p_level3[y*width/8+x] = sigma/coeff;		
	}
	
	/**************************************************************************/
	
	/*** Open Octree Poisson ***/
	void cu_SolveOpenOctreePoissonRedBlack_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, 
											const int width, const int height, const int maxIter)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x, (height+1+blockSize.y-1)/blockSize.y);
		
		
		int levelWidth1 = width/2;
		int levelWidth2 = width/4;
		int levelWidth3 = width/8;
		int levelHeight1 = height/2;
		int levelHeight2 = height/4;
		int levelHeight3 = height/8;
		
		dim3 gridSize1((levelWidth1+blockSize.x-1)/blockSize.x, (levelHeight1+blockSize.y-1)/blockSize.y);
		dim3 gridSize2((levelWidth2+blockSize.x-1)/blockSize.x, (levelHeight2+blockSize.y-1)/blockSize.y);
		dim3 gridSize3((levelWidth3+blockSize.x-1)/blockSize.x, (levelHeight3+blockSize.y-1)/blockSize.y);
		
		float* divergence0 = 0;
		float* divergence1 = 0;
		float* divergence2 = 0;
		float* divergence3 = 0;
		checkCudaErrors( cudaMalloc((void**)&divergence0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&divergence1,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMalloc((void**)&divergence2,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMalloc((void**)&divergence3,sizeof(float)*levelWidth3*levelHeight3) );
		checkCudaErrors( cudaMemset(divergence0,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(divergence1,0,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMemset(divergence2,0,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMemset(divergence3,0,sizeof(float)*levelWidth3*levelHeight3) );
		
		float* p0 = 0;
		float* p1 = 0;
		float* p2 = 0;
		float* p3 = 0;
		checkCudaErrors( cudaMalloc((void**)&p0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&p1,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMalloc((void**)&p2,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMalloc((void**)&p3,sizeof(float)*levelWidth3*levelHeight3) );
		checkCudaErrors( cudaMemset(p0,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(p1,0,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMemset(p2,0,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMemset(p3,0,sizeof(float)*levelWidth3*levelHeight3) );
		
		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(divergence0,mac_u,mac_v,width,height);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize1,blockSize>>>(divergence1,divergence0,levelWidth1,levelHeight1);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize2,blockSize>>>(divergence2,divergence1,levelWidth2,levelHeight2);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize3,blockSize>>>(divergence3,divergence2,levelWidth3,levelHeight3);
		
		for(int it = 0;it < maxIter;it++)
		{
			SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel<<<gridSize,blockSize>>>(p0, p1, p2, p3, divergence0, leaf0, leaf1, leaf2, leaf3, width, height, true);
			SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel<<<gridSize,blockSize>>>(p0, p1, p2, p3, divergence0, leaf0, leaf1, leaf2, leaf3, width, height, false);
			SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel<<<gridSize1,blockSize>>>(p0, p1, p2, p3, divergence1, leaf0, leaf1, leaf2, leaf3, width, height, true);
			SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel<<<gridSize1,blockSize>>>(p0, p1, p2, p3, divergence1, leaf0, leaf1, leaf2, leaf3, width, height, false);
			SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel<<<gridSize2,blockSize>>>(p0, p1, p2, p3, divergence2, leaf0, leaf1, leaf2, leaf3, width, height, true);
			SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel<<<gridSize2,blockSize>>>(p0, p1, p2, p3, divergence2, leaf0, leaf1, leaf2, leaf3, width, height, false);
			SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel<<<gridSize3,blockSize>>>(p0, p1, p2, p3, divergence3, leaf0, leaf1, leaf2, leaf3, width, height, true);
			SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel<<<gridSize3,blockSize>>>(p0, p1, p2, p3, divergence3, leaf0, leaf1, leaf2, leaf3, width, height, false);
		}
		
		Adjust_MAC_u_OpenOctreePoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u, p0, p1, p2, p3, leaf0, leaf1, leaf2, leaf3, width, height);
		Adjust_MAC_v_OpenOctreePoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v, p0, p1, p2, p3, leaf0, leaf1, leaf2, leaf3, width, height);
		
		checkCudaErrors( cudaFree(divergence0) );
		checkCudaErrors( cudaFree(divergence1) );
		checkCudaErrors( cudaFree(divergence2) );
		checkCudaErrors( cudaFree(divergence3) );
		checkCudaErrors( cudaFree(p0) );
		checkCudaErrors( cudaFree(p1) );
		checkCudaErrors( cudaFree(p2) );
		checkCudaErrors( cudaFree(p3) );
		divergence0 = 0;
		divergence1 = 0;
		divergence2 = 0;
		divergence3 = 0;
		p0 = 0;
		p1 = 0;
		p2 = 0;
		p3 = 0;
	}
	
	void cu_SolveOpenOctreePoissonRedBlack_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, 
											const int width, const int height, const int maxIter,
											const int level0_num_red, const int* level0_index_red, const int level0_num_black, const int* level0_index_black,
											const int level1_num_red, const int* level1_index_red, const int level1_num_black, const int* level1_index_black,
											const int level2_num_red, const int* level2_index_red, const int level2_num_black, const int* level2_index_black,
											const int level3_num_red, const int* level3_index_red, const int level3_num_black, const int* level3_index_black)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x, (height+1+blockSize.y-1)/blockSize.y);
		
		
		int levelWidth1 = width/2;
		int levelWidth2 = width/4;
		int levelWidth3 = width/8;
		int levelHeight1 = height/2;
		int levelHeight2 = height/4;
		int levelHeight3 = height/8;
		
		dim3 gridSize1((levelWidth1+blockSize.x-1)/blockSize.x, (levelHeight1+blockSize.y-1)/blockSize.y);
		dim3 gridSize2((levelWidth2+blockSize.x-1)/blockSize.x, (levelHeight2+blockSize.y-1)/blockSize.y);
		dim3 gridSize3((levelWidth3+blockSize.x-1)/blockSize.x, (levelHeight3+blockSize.y-1)/blockSize.y);
		
		float* divergence0 = 0;
		float* divergence1 = 0;
		float* divergence2 = 0;
		float* divergence3 = 0;
		checkCudaErrors( cudaMalloc((void**)&divergence0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&divergence1,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMalloc((void**)&divergence2,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMalloc((void**)&divergence3,sizeof(float)*levelWidth3*levelHeight3) );
		checkCudaErrors( cudaMemset(divergence0,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(divergence1,0,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMemset(divergence2,0,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMemset(divergence3,0,sizeof(float)*levelWidth3*levelHeight3) );
		
		float* p0 = 0;
		float* p1 = 0;
		float* p2 = 0;
		float* p3 = 0;
		checkCudaErrors( cudaMalloc((void**)&p0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&p1,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMalloc((void**)&p2,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMalloc((void**)&p3,sizeof(float)*levelWidth3*levelHeight3) );
		checkCudaErrors( cudaMemset(p0,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(p1,0,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMemset(p2,0,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMemset(p3,0,sizeof(float)*levelWidth3*levelHeight3) );
		
		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(divergence0,mac_u,mac_v,width,height);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize1,blockSize>>>(divergence1,divergence0,levelWidth1,levelHeight1);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize2,blockSize>>>(divergence2,divergence1,levelWidth2,levelHeight2);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize3,blockSize>>>(divergence3,divergence2,levelWidth3,levelHeight3);
		
		dim3 len_blockSize(64,1);
		dim3 level0_red_gridSize((level0_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level0_black_gridSize((level0_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level1_red_gridSize((level1_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level1_black_gridSize((level1_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level2_red_gridSize((level2_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level2_black_gridSize((level2_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level3_red_gridSize((level3_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level3_black_gridSize((level3_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		
		for(int it = 0;it < maxIter;it++)
		{
			if(level0_num_red > 0)
				SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel<<<level0_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence0, leaf0, leaf1, leaf2, leaf3, width, height, level0_num_red,level0_index_red);
			if(level0_num_black > 0)
				SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel<<<level0_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence0, leaf0, leaf1, leaf2, leaf3, width, height, level0_num_black,level0_index_black);
			
			if(level1_num_red > 0)
				SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel<<<level1_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence1, leaf0, leaf1, leaf2, leaf3, width, height, level1_num_red,level1_index_red);

			if(level1_num_black > 0)
				SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel<<<level1_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence1, leaf0, leaf1, leaf2, leaf3, width, height, level1_num_black,level1_index_black);
						
			if(level2_num_red > 0)
				SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel<<<level2_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence2, leaf0, leaf1, leaf2, leaf3, width, height, level2_num_red,level2_index_red);

			if(level2_num_black > 0)
				SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel<<<level2_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence2, leaf0, leaf1, leaf2, leaf3, width, height, level2_num_black,level2_index_black);	
						
			if(level3_num_red > 0)
				SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel<<<level3_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence3, leaf0, leaf1, leaf2, leaf3, width, height, level3_num_red,level3_index_red);

			if(level3_num_black > 0)
				SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel<<<level3_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence3, leaf0, leaf1, leaf2, leaf3, width, height, level3_num_black,level3_index_black);			
						
		}
		
		Adjust_MAC_u_OpenOctreePoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u, p0, p1, p2, p3, leaf0, leaf1, leaf2, leaf3, width, height);
		Adjust_MAC_v_OpenOctreePoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v, p0, p1, p2, p3, leaf0, leaf1, leaf2, leaf3, width, height);
		
		checkCudaErrors( cudaFree(divergence0) );
		checkCudaErrors( cudaFree(divergence1) );
		checkCudaErrors( cudaFree(divergence2) );
		checkCudaErrors( cudaFree(divergence3) );
		checkCudaErrors( cudaFree(p0) );
		checkCudaErrors( cudaFree(p1) );
		checkCudaErrors( cudaFree(p2) );
		checkCudaErrors( cudaFree(p3) );
		divergence0 = 0;
		divergence1 = 0;
		divergence2 = 0;
		divergence3 = 0;
		p0 = 0;
		p1 = 0;
		p2 = 0;
		p3 = 0;	
	}
	
	void cu_SolveOpenOctreePoissonRedBlack_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3, 
		const int width, const int height, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int* level0_neighborinfo_red,
		const int level0_num_black, const int* level0_index_black, const int* level0_neighborinfo_black,
		const int level1_num_red, const int* level1_index_red, const int* level1_neighborinfo_red, 
		const int level1_num_black, const int* level1_index_black, const int* level1_neighborinfo_black,
		const int level2_num_red, const int* level2_index_red, const int* level2_neighborinfo_red,
		const int level2_num_black, const int* level2_index_black, const int* level2_neighborinfo_black,
		const int level3_num_red, const int* level3_index_red, const int* level3_neighborinfo_red,
		const int level3_num_black, const int* level3_index_black, const int* level3_neighborinfo_black)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
		dim3 u_gridSize((width+1+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
		dim3 v_gridSize((width+blockSize.x-1)/blockSize.x, (height+1+blockSize.y-1)/blockSize.y);
		
		
		int levelWidth1 = width/2;
		int levelWidth2 = width/4;
		int levelWidth3 = width/8;
		int levelHeight1 = height/2;
		int levelHeight2 = height/4;
		int levelHeight3 = height/8;
		
		dim3 gridSize1((levelWidth1+blockSize.x-1)/blockSize.x, (levelHeight1+blockSize.y-1)/blockSize.y);
		dim3 gridSize2((levelWidth2+blockSize.x-1)/blockSize.x, (levelHeight2+blockSize.y-1)/blockSize.y);
		dim3 gridSize3((levelWidth3+blockSize.x-1)/blockSize.x, (levelHeight3+blockSize.y-1)/blockSize.y);
		
		float* divergence0 = 0;
		float* divergence1 = 0;
		float* divergence2 = 0;
		float* divergence3 = 0;
		checkCudaErrors( cudaMalloc((void**)&divergence0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&divergence1,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMalloc((void**)&divergence2,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMalloc((void**)&divergence3,sizeof(float)*levelWidth3*levelHeight3) );
		checkCudaErrors( cudaMemset(divergence0,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(divergence1,0,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMemset(divergence2,0,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMemset(divergence3,0,sizeof(float)*levelWidth3*levelHeight3) );
		
		float* p0 = 0;
		float* p1 = 0;
		float* p2 = 0;
		float* p3 = 0;
		checkCudaErrors( cudaMalloc((void**)&p0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&p1,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMalloc((void**)&p2,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMalloc((void**)&p3,sizeof(float)*levelWidth3*levelHeight3) );
		checkCudaErrors( cudaMemset(p0,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(p1,0,sizeof(float)*levelWidth1*levelHeight1) );
		checkCudaErrors( cudaMemset(p2,0,sizeof(float)*levelWidth2*levelHeight2) );
		checkCudaErrors( cudaMemset(p3,0,sizeof(float)*levelWidth3*levelHeight3) );
		
		Calculate_Divergence_of_MAC_Kernel<<<gridSize,blockSize>>>(divergence0,mac_u,mac_v,width,height);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize1,blockSize>>>(divergence1,divergence0,levelWidth1,levelHeight1);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize2,blockSize>>>(divergence2,divergence1,levelWidth2,levelHeight2);
		Calculate_Divergence_Octree_from_previous_level_Kernel<<<gridSize3,blockSize>>>(divergence3,divergence2,levelWidth3,levelHeight3);
		
		dim3 len_blockSize(64,1);
		dim3 level0_red_gridSize((level0_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level0_black_gridSize((level0_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level1_red_gridSize((level1_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level1_black_gridSize((level1_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level2_red_gridSize((level2_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level2_black_gridSize((level2_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level3_red_gridSize((level3_num_red+len_blockSize.x-1)/len_blockSize.x,1);
		dim3 level3_black_gridSize((level3_num_black+len_blockSize.x-1)/len_blockSize.x,1);
		
		for(int it = 0;it < maxIter;it++)
		{
			if(level0_num_red > 0)
				SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel<<<level0_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence0, width, height, level0_num_red,level0_index_red,level0_neighborinfo_red);
			if(level0_num_black > 0)
				SolvePressure_OpenOctreePoisson_level0_RedBlack_Kernel<<<level0_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence0, width, height, level0_num_black,level0_index_black,level0_neighborinfo_black);
			
			if(level1_num_red > 0)
				SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel<<<level1_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence1, width, height, level1_num_red,level1_index_red,level1_neighborinfo_red);

			if(level1_num_black > 0)
				SolvePressure_OpenOctreePoisson_level1_RedBlack_Kernel<<<level1_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence1, width, height, level1_num_black,level1_index_black,level1_neighborinfo_black);
						
			if(level2_num_red > 0)
				SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel<<<level2_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence2, width, height, level2_num_red,level2_index_red,level2_neighborinfo_red);

			if(level2_num_black > 0)
				SolvePressure_OpenOctreePoisson_level2_RedBlack_Kernel<<<level2_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence2, width, height, level2_num_black,level2_index_black,level2_neighborinfo_black);	
						
			if(level3_num_red > 0)
				SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel<<<level3_red_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence3, width, height, level3_num_red,level3_index_red,level3_neighborinfo_red);

			if(level3_num_black > 0)
				SolvePressure_OpenOctreePoisson_level3_RedBlack_Kernel<<<level3_black_gridSize,len_blockSize>>>(
						p0, p1, p2, p3, divergence3, width, height, level3_num_black,level3_index_black,level3_neighborinfo_black);			
						
		}
		
		Adjust_MAC_u_OpenOctreePoisson_Kernel<<<u_gridSize,blockSize>>>(mac_u, p0, p1, p2, p3, leaf0, leaf1, leaf2, leaf3, width, height);
		Adjust_MAC_v_OpenOctreePoisson_Kernel<<<v_gridSize,blockSize>>>(mac_v, p0, p1, p2, p3, leaf0, leaf1, leaf2, leaf3, width, height);
		
		checkCudaErrors( cudaFree(divergence0) );
		checkCudaErrors( cudaFree(divergence1) );
		checkCudaErrors( cudaFree(divergence2) );
		checkCudaErrors( cudaFree(divergence3) );
		checkCudaErrors( cudaFree(p0) );
		checkCudaErrors( cudaFree(p1) );
		checkCudaErrors( cudaFree(p2) );
		checkCudaErrors( cudaFree(p3) );
		divergence0 = 0;
		divergence1 = 0;
		divergence2 = 0;
		divergence3 = 0;
		p0 = 0;
		p1 = 0;
		p2 = 0;
		p3 = 0;	
	}
	
	/***************************************************************************/
	
	/*** Open Octree Poisson ***/
	extern "C"
	void SolveOpenOctreePoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
											const int width, const int height, const int maxIter)
	{
		if(width%8 != 0 || height%8 != 0 || width < 24 || height < 24)
		{
			printf("invalid resolution [%d x %d]\n",width,height);
			return ;
		}
		
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* leaf0_d = 0;
		bool* leaf1_d = 0;
		bool* leaf2_d = 0;
		bool* leaf3_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&leaf0_d,sizeof(bool)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&leaf1_d,sizeof(bool)*width/2*height/2) );
		checkCudaErrors( cudaMalloc((void**)&leaf2_d,sizeof(bool)*width/4*height/4) );
		checkCudaErrors( cudaMalloc((void**)&leaf3_d,sizeof(bool)*width/8*height/8) );
		
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf0_d,leaf0,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf1_d,leaf1,sizeof(bool)*width/2*height/2,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf2_d,leaf2,sizeof(bool)*width/4*height/4,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf3_d,leaf3,sizeof(bool)*width/8*height/8,cudaMemcpyHostToDevice) );
		
		cu_SolveOpenOctreePoissonRedBlack_MAC(mac_u_d, mac_v_d, leaf0_d, leaf1_d, leaf2_d, leaf3_d, width, height, maxIter);
		
		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(leaf0_d) );
		checkCudaErrors( cudaFree(leaf1_d) );
		checkCudaErrors( cudaFree(leaf2_d) );
		checkCudaErrors( cudaFree(leaf3_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		leaf0_d = 0;
		leaf1_d = 0;
		leaf2_d = 0;
		leaf3_d = 0;
	}
	
	extern "C"
	void SolveOpenOctreePoissonRedBlack2_2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
											const int width, const int height, const int maxIter,
											const int level0_num_red, const int* level0_index_red, const int level0_num_black, const int* level0_index_black,
											const int level1_num_red, const int* level1_index_red, const int level1_num_black, const int* level1_index_black,
											const int level2_num_red, const int* level2_index_red, const int level2_num_black, const int* level2_index_black,
											const int level3_num_red, const int* level3_index_red, const int level3_num_black, const int* level3_index_black)
	{
		if(width%8 != 0 || height%8 != 0 || width < 24 || height < 24)
		{
			printf("invalid resolution [%d x %d]\n",width,height);
			return ;
		}
		
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* leaf0_d = 0;
		bool* leaf1_d = 0;
		bool* leaf2_d = 0;
		bool* leaf3_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&leaf0_d,sizeof(bool)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&leaf1_d,sizeof(bool)*width/2*height/2) );
		checkCudaErrors( cudaMalloc((void**)&leaf2_d,sizeof(bool)*width/4*height/4) );
		checkCudaErrors( cudaMalloc((void**)&leaf3_d,sizeof(bool)*width/8*height/8) );
		
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf0_d,leaf0,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf1_d,leaf1,sizeof(bool)*width/2*height/2,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf2_d,leaf2,sizeof(bool)*width/4*height/4,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf3_d,leaf3,sizeof(bool)*width/8*height/8,cudaMemcpyHostToDevice) );
		
		int* level0_index_red_d = 0;
		int* level0_index_black_d = 0;
		int* level1_index_red_d = 0;
		int* level1_index_black_d = 0;
		int* level2_index_red_d = 0;
		int* level2_index_black_d = 0;
		int* level3_index_red_d = 0;
		int* level3_index_black_d = 0;
		
		if(level0_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level0_index_red_d,sizeof(int)*level0_num_red*2) );
			checkCudaErrors( cudaMemcpy(level0_index_red_d,level0_index_red,sizeof(int)*level0_num_red*2,cudaMemcpyHostToDevice) );
		}
		
		if(level0_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level0_index_black_d,sizeof(int)*level0_num_black*2) );
			checkCudaErrors( cudaMemcpy(level0_index_black_d,level0_index_black,sizeof(int)*level0_num_black*2,cudaMemcpyHostToDevice) );
		}
		
		if(level1_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level1_index_red_d,sizeof(int)*level1_num_red*2) );
			checkCudaErrors( cudaMemcpy(level1_index_red_d,level1_index_red,sizeof(int)*level1_num_red*2,cudaMemcpyHostToDevice) );
		}
		
		if(level1_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level1_index_black_d,sizeof(int)*level1_num_black*2) );
			checkCudaErrors( cudaMemcpy(level1_index_black_d,level1_index_black,sizeof(int)*level1_num_black*2,cudaMemcpyHostToDevice) );
		}
		
		if(level2_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level2_index_red_d,sizeof(int)*level2_num_red*2) );
			checkCudaErrors( cudaMemcpy(level2_index_red_d,level2_index_red,sizeof(int)*level2_num_red*2,cudaMemcpyHostToDevice) );
		}
		
		if(level2_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level2_index_black_d,sizeof(int)*level2_num_black*2) );
			checkCudaErrors( cudaMemcpy(level2_index_black_d,level2_index_black,sizeof(int)*level2_num_black*2,cudaMemcpyHostToDevice) );
		}
		
		if(level3_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level3_index_red_d,sizeof(int)*level3_num_red*2) );
			checkCudaErrors( cudaMemcpy(level3_index_red_d,level3_index_red,sizeof(int)*level3_num_red*2,cudaMemcpyHostToDevice) );
		}
		
		if(level3_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level3_index_black_d,sizeof(int)*level3_num_black*2) );
			checkCudaErrors( cudaMemcpy(level3_index_black_d,level3_index_black,sizeof(int)*level3_num_black*2,cudaMemcpyHostToDevice) );
		}
		
		cu_SolveOpenOctreePoissonRedBlack_MAC(mac_u_d, mac_v_d, leaf0_d, leaf1_d, leaf2_d, leaf3_d, width, height, maxIter,
										level0_num_red,level0_index_red_d,level0_num_black,level0_index_black_d,
										level1_num_red,level1_index_red_d,level1_num_black,level1_index_black_d,
										level2_num_red,level2_index_red_d,level2_num_black,level2_index_black_d,
										level3_num_red,level3_index_red_d,level3_num_black,level3_index_black_d);
		
		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(leaf0_d) );
		checkCudaErrors( cudaFree(leaf1_d) );
		checkCudaErrors( cudaFree(leaf2_d) );
		checkCudaErrors( cudaFree(leaf3_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		leaf0_d = 0;
		leaf1_d = 0;
		leaf2_d = 0;
		leaf3_d = 0;
		
		if(level0_index_red_d)
		{
			checkCudaErrors( cudaFree(level0_index_red_d) );
			level0_index_red_d = 0;
		}
		if(level0_index_black_d)
		{
			checkCudaErrors( cudaFree(level0_index_black_d) );
			level0_index_black_d = 0;
		}
		if(level1_index_red_d)
		{
			checkCudaErrors( cudaFree(level1_index_red_d) );
			level1_index_red_d = 0;
		}
		if(level1_index_black_d)
		{
			checkCudaErrors( cudaFree(level1_index_black_d) );
			level1_index_black_d = 0;
		}
		if(level2_index_red_d)
		{
			checkCudaErrors( cudaFree(level2_index_red_d) );
			level2_index_red_d = 0;
		}
		if(level2_index_black_d)
		{
			checkCudaErrors( cudaFree(level2_index_black_d) );
			level2_index_black_d = 0;
		}
		if(level3_index_red_d)
		{
			checkCudaErrors( cudaFree(level3_index_red_d) );
			level3_index_red_d = 0;
		}
		if(level3_index_black_d)
		{
			checkCudaErrors( cudaFree(level3_index_black_d) );
			level3_index_black_d = 0;
		}
	}
	
	/*index: [x,y,num,offset]...
	* neighbor info : [level,x,y]...
	*/
	extern "C"
	void SolveOpenOctreePoissonRedBlack3_2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const int width, const int height, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int level0_info_len_red, const int* level0_neighborinfo_red,
		const int level0_num_black, const int* level0_index_black, const int level0_info_len_black, const int* level0_neighborinfo_black,
		const int level1_num_red, const int* level1_index_red, const int level1_info_len_red, const int* level1_neighborinfo_red, 
		const int level1_num_black, const int* level1_index_black, const int level1_info_len_black, const int* level1_neighborinfo_black,
		const int level2_num_red, const int* level2_index_red, const int level2_info_len_red, const int* level2_neighborinfo_red,
		const int level2_num_black, const int* level2_index_black, const int level2_info_len_black, const int* level2_neighborinfo_black,
		const int level3_num_red, const int* level3_index_red, const int level3_info_len_red, const int* level3_neighborinfo_red,
		const int level3_num_black, const int* level3_index_black, const int level3_info_len_black, const int* level3_neighborinfo_black)
	{
		if(width%8 != 0 || height%8 != 0 || width < 24 || height < 24)
		{
			printf("invalid resolution [%d x %d]\n",width,height);
			return ;
		}
		
		float* mac_u_d = 0;
		float* mac_v_d = 0;
		bool* leaf0_d = 0;
		bool* leaf1_d = 0;
		bool* leaf2_d = 0;
		bool* leaf3_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&mac_u_d,sizeof(float)*(width+1)*height) );
		checkCudaErrors( cudaMalloc((void**)&mac_v_d,sizeof(float)*width*(height+1)) );
		checkCudaErrors( cudaMalloc((void**)&leaf0_d,sizeof(bool)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&leaf1_d,sizeof(bool)*width/2*height/2) );
		checkCudaErrors( cudaMalloc((void**)&leaf2_d,sizeof(bool)*width/4*height/4) );
		checkCudaErrors( cudaMalloc((void**)&leaf3_d,sizeof(bool)*width/8*height/8) );
		
		checkCudaErrors( cudaMemcpy(mac_u_d,mac_u,sizeof(float)*(width+1)*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(mac_v_d,mac_v,sizeof(float)*width*(height+1),cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf0_d,leaf0,sizeof(bool)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf1_d,leaf1,sizeof(bool)*width/2*height/2,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf2_d,leaf2,sizeof(bool)*width/4*height/4,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(leaf3_d,leaf3,sizeof(bool)*width/8*height/8,cudaMemcpyHostToDevice) );
		
		int* level0_index_red_d = 0;
		int* level0_neighborinfo_red_d = 0;
		int* level0_index_black_d = 0;
		int* level0_neighborinfo_black_d = 0;
		int* level1_index_red_d = 0;
		int* level1_neighborinfo_red_d = 0;
		int* level1_index_black_d = 0;
		int* level1_neighborinfo_black_d = 0;
		int* level2_index_red_d = 0;
		int* level2_neighborinfo_red_d = 0;
		int* level2_index_black_d = 0;
		int* level2_neighborinfo_black_d = 0;
		int* level3_index_red_d = 0;
		int* level3_neighborinfo_red_d = 0;
		int* level3_index_black_d = 0;
		int* level3_neighborinfo_black_d = 0;
		
		const int index_channels = 4;
		const int neighborinfo_channels = 3;
		if(level0_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level0_index_red_d,sizeof(int)*level0_num_red*index_channels) );
			checkCudaErrors( cudaMemcpy(level0_index_red_d,level0_index_red,sizeof(int)*level0_num_red*index_channels,cudaMemcpyHostToDevice) );
			if(level0_info_len_red > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level0_neighborinfo_red_d,sizeof(int)*level0_info_len_red*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level0_neighborinfo_red_d,level0_neighborinfo_red,sizeof(int)*level0_info_len_red*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		if(level0_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level0_index_black_d,sizeof(int)*level0_num_black*index_channels) );
			checkCudaErrors( cudaMemcpy(level0_index_black_d,level0_index_black,sizeof(int)*level0_num_black*index_channels,cudaMemcpyHostToDevice) );
			if(level0_info_len_black > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level0_neighborinfo_black_d,sizeof(int)*level0_info_len_black*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level0_neighborinfo_black_d,level0_neighborinfo_black,sizeof(int)*level0_info_len_black*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}	
		}
		
		if(level1_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level1_index_red_d,sizeof(int)*level1_num_red*index_channels) );
			checkCudaErrors( cudaMemcpy(level1_index_red_d,level1_index_red,sizeof(int)*level1_num_red*index_channels,cudaMemcpyHostToDevice) );
			if(level1_info_len_red > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level1_neighborinfo_red_d,sizeof(int)*level1_info_len_red*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level1_neighborinfo_red_d,level1_neighborinfo_red,sizeof(int)*level1_info_len_red*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		if(level1_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level1_index_black_d,sizeof(int)*level1_num_black*index_channels) );
			checkCudaErrors( cudaMemcpy(level1_index_black_d,level1_index_black,sizeof(int)*level1_num_black*index_channels,cudaMemcpyHostToDevice) );
			if(level1_info_len_black > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level1_neighborinfo_black_d,sizeof(int)*level1_info_len_black*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level1_neighborinfo_black_d,level1_neighborinfo_black,sizeof(int)*level1_info_len_black*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		if(level2_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level2_index_red_d,sizeof(int)*level2_num_red*index_channels) );
			checkCudaErrors( cudaMemcpy(level2_index_red_d,level2_index_red,sizeof(int)*level2_num_red*index_channels,cudaMemcpyHostToDevice) );
			if(level2_info_len_red > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level2_neighborinfo_red_d,sizeof(int)*level2_info_len_red*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level2_neighborinfo_red_d,level2_neighborinfo_red,sizeof(int)*level2_info_len_red*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		if(level2_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level2_index_black_d,sizeof(int)*level2_num_black*index_channels) );
			checkCudaErrors( cudaMemcpy(level2_index_black_d,level2_index_black,sizeof(int)*level2_num_black*index_channels,cudaMemcpyHostToDevice) );
			if(level2_info_len_black > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level2_neighborinfo_black_d,sizeof(int)*level2_info_len_black*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level2_neighborinfo_black_d,level2_neighborinfo_black,sizeof(int)*level2_info_len_black*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		if(level3_num_red > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level3_index_red_d,sizeof(int)*level3_num_red*index_channels) );
			checkCudaErrors( cudaMemcpy(level3_index_red_d,level3_index_red,sizeof(int)*level3_num_red*index_channels,cudaMemcpyHostToDevice) );
			if(level3_info_len_red > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level3_neighborinfo_red_d,sizeof(int)*level3_info_len_red*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level3_neighborinfo_red_d,level3_neighborinfo_red,sizeof(int)*level3_info_len_red*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		if(level3_num_black > 0)
		{
			checkCudaErrors( cudaMalloc((void**)&level3_index_black_d,sizeof(int)*level3_num_black*index_channels) );
			checkCudaErrors( cudaMemcpy(level3_index_black_d,level3_index_black,sizeof(int)*level3_num_black*index_channels,cudaMemcpyHostToDevice) );
			if(level3_info_len_black > 0)
			{
				checkCudaErrors( cudaMalloc((void**)&level3_neighborinfo_black_d,sizeof(int)*level3_info_len_black*neighborinfo_channels) );
				checkCudaErrors( cudaMemcpy(level3_neighborinfo_black_d,level3_neighborinfo_black,sizeof(int)*level3_info_len_black*neighborinfo_channels,cudaMemcpyHostToDevice) );
			}
		}
		
		cu_SolveOpenOctreePoissonRedBlack_MAC(mac_u_d, mac_v_d, leaf0_d, leaf1_d, leaf2_d, leaf3_d, width, height, maxIter,
										level0_num_red,level0_index_red_d,level0_neighborinfo_red_d,
										level0_num_black,level0_index_black_d,level0_neighborinfo_black_d,
										level1_num_red,level1_index_red_d,level1_neighborinfo_red_d,
										level1_num_black,level1_index_black_d,level1_neighborinfo_black_d,
										level2_num_red,level2_index_red_d,level2_neighborinfo_red_d,
										level2_num_black,level2_index_black_d,level2_neighborinfo_black_d,
										level3_num_red,level3_index_red_d,level3_neighborinfo_red_d,
										level3_num_black,level3_index_black_d,level3_neighborinfo_black_d);
		
		checkCudaErrors( cudaMemcpy(mac_u,mac_u_d,sizeof(float)*(width+1)*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(mac_v,mac_v_d,sizeof(float)*width*(height+1),cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(mac_u_d) );
		checkCudaErrors( cudaFree(mac_v_d) );
		checkCudaErrors( cudaFree(leaf0_d) );
		checkCudaErrors( cudaFree(leaf1_d) );
		checkCudaErrors( cudaFree(leaf2_d) );
		checkCudaErrors( cudaFree(leaf3_d) );
		mac_u_d = 0;
		mac_v_d = 0;
		leaf0_d = 0;
		leaf1_d = 0;
		leaf2_d = 0;
		leaf3_d = 0;
		
		if(level0_index_red_d)
		{
			checkCudaErrors( cudaFree(level0_index_red_d) );
			level0_index_red_d = 0;
		}
		if(level0_index_black_d)
		{
			checkCudaErrors( cudaFree(level0_index_black_d) );
			level0_index_black_d = 0;
		}
		if(level1_index_red_d)
		{
			checkCudaErrors( cudaFree(level1_index_red_d) );
			level1_index_red_d = 0;
		}
		if(level1_index_black_d)
		{
			checkCudaErrors( cudaFree(level1_index_black_d) );
			level1_index_black_d = 0;
		}
		if(level2_index_red_d)
		{
			checkCudaErrors( cudaFree(level2_index_red_d) );
			level2_index_red_d = 0;
		}
		if(level2_index_black_d)
		{
			checkCudaErrors( cudaFree(level2_index_black_d) );
			level2_index_black_d = 0;
		}
		if(level3_index_red_d)
		{
			checkCudaErrors( cudaFree(level3_index_red_d) );
			level3_index_red_d = 0;
		}
		if(level3_index_black_d)
		{
			checkCudaErrors( cudaFree(level3_index_black_d) );
			level3_index_black_d = 0;
		}
		
		if(level0_neighborinfo_red_d)
		{
			checkCudaErrors( cudaFree(level0_neighborinfo_red_d) );
			level0_neighborinfo_red_d = 0;
		}
		if(level0_neighborinfo_black_d)
		{
			checkCudaErrors( cudaFree(level0_neighborinfo_black_d) );
			level0_neighborinfo_black_d = 0;
		}
		if(level1_neighborinfo_red_d)
		{
			checkCudaErrors( cudaFree(level1_neighborinfo_red_d) );
			level1_neighborinfo_red_d = 0;
		}
		if(level1_neighborinfo_black_d)
		{
			checkCudaErrors( cudaFree(level1_neighborinfo_black_d) );
			level1_neighborinfo_black_d = 0;
		}
		if(level2_neighborinfo_red_d)
		{
			checkCudaErrors( cudaFree(level2_neighborinfo_red_d) );
			level2_neighborinfo_red_d = 0;
		}
		if(level2_neighborinfo_black_d)
		{
			checkCudaErrors( cudaFree(level2_neighborinfo_black_d) );
			level2_neighborinfo_black_d = 0;
		}
		if(level3_neighborinfo_red_d)
		{
			checkCudaErrors( cudaFree(level3_neighborinfo_red_d) );
			level3_neighborinfo_red_d = 0;
		}
		if(level3_neighborinfo_black_d)
		{
			checkCudaErrors( cudaFree(level3_neighborinfo_black_d) );
			level3_neighborinfo_black_d = 0;
		}
	}
}


#endif