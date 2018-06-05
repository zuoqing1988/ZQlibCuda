#ifndef _ZQ_CUDA_BLEND_TWO_IMAGES3D_CU_
#define _ZQ_CUDA_BLEND_TWO_IMAGES3D_CU_


#include "ZQ_CUDA_BaseUtils.cuh"
#include "ZQ_CUDA_BlendTwoImages3D.cuh"

#define MAX_NEIGHBOR_NUM 64

#define _kernel_square wendland_square

namespace ZQ_CUDA_BlendTwoImages3D
{


	/******************************************/

	__device__
	float gaussian(const float dis, const float radius)
	{
		if(fabs(dis) > fabs(radius))
			return 0;
		float d = dis / radius;
		return exp(-d*d*6);
	}
	
	__device__
	float gaussian_square(const float dis2, const float radius2)
	{
		//if(dis2 > radius2)
		//	return 0;
		float d = dis2 / radius2;
		return exp(-d*6);
	}
	
	__device__
	float wendland_square(const float dis2, const float radius2)
	{
		float d2 = dis2/radius2;
		float d = sqrt(d2);
		float tmp = 1.0 - d;
		tmp *= tmp;
		tmp *= tmp;
		return tmp*(4.0*d+1.0);
	}

	/*input = {I(z1,y0,x0),I(z1,y0,x1),I(z1,y1,x0),I(z1,y1,x1),I(z1,y0,x0),I(z1,y0,x1),I(z1,y1,x0),I(z1,y1,x1)}, 
	x \in [0,1], y \in [0,1], z \in [0,1]*/
	__device__  
	float trilinear_interpolation(const float* input, const float x, const float y, const float z)
	{
		float w000 = (1-z)*(1-y)*(1-x);
		float w001 = (1-z)*(1-y)*x;
		float w010 = (1-z)*y*(1-x);
		float w011 = (1-z)*y*x;
		float w100 = z*(1-y)*(1-x);
		float w101 = z*(1-y)*x;
		float w110 = z*y*(1-x);
		float w111 = z*y*x;

		return input[0]*w000 + input[1]*w001 + input[2]*w010 + input[3]*w011
			  +input[4]*w100 + input[5]*w101 + input[6]*w110 + input[7]*w111;
	}

	__global__
	void compute_coord_in_which_bucket_kernel(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float x_min, const float y_min, const float z_min,
								const float radius, const int bucket_width, const int bucket_height, const int bucket_depth, int* coord_in_which_bucket)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx >= num)
			return ;
			
		int d_idx = (coord_z[idx] - z_min)/radius;
		int h_idx = (coord_y[idx] - y_min)/radius;
		int w_idx = (coord_x[idx] - x_min)/radius;
			
		d_idx = __max(0,__min(bucket_depth-1,d_idx));
		h_idx = __max(0,__min(bucket_height-1,h_idx));
		w_idx = __max(0,__min(bucket_width-1,w_idx));
		int bucket_idx = d_idx*bucket_height*bucket_width+h_idx*bucket_width+w_idx;
		coord_in_which_bucket[idx] = bucket_idx;
	}
	
	__global__
	void Compute_Neightbors_kernel(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float radius, 
					const int bucket_width, const int bucket_height, const int bucket_depth,
					const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
					int* neighbor_num, int* neighbor_index, float* neighbor_weight)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num)
			return;

		int bucket_idx = coord_in_which_bucket[idx];
		
		int bucket_slice = bucket_width*bucket_height;
		int d_idx = bucket_idx/bucket_slice;
		int rest_idx = bucket_idx%bucket_slice;
		int w_idx = rest_idx%bucket_width;
		int h_idx = rest_idx/bucket_width;

		float cur_x = coord_x[idx];
		float cur_y = coord_y[idx];
		float cur_z = coord_z[idx];
		int cur_neigh_num = 0;
		
		float radius2 = radius*radius;
		
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{ 
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				for(int cur_d_idx = __max(0,d_idx-1);cur_d_idx <= __min(bucket_depth-1,d_idx+1);cur_d_idx++)
				{
					int cur_bucket_idx = cur_d_idx*bucket_slice+cur_h_idx*bucket_width+cur_w_idx;
					int cur_offset = bucket_stored_offset[cur_bucket_idx];
					for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
					{
						int cur_nei_idx = bucket_stored_index[cur_offset+iii];
						if(cur_nei_idx == idx)
							continue;
						float cur_nei_x = coord_x[cur_nei_idx];
						float cur_nei_y = coord_y[cur_nei_idx];
						float cur_nei_z = coord_z[cur_nei_idx];
						float cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y)+(cur_z-cur_nei_z)*(cur_z-cur_nei_z);
						if(cur_dis2 <= radius2)
						{
							if(cur_neigh_num < MAX_NEIGHBOR_NUM)
							{
								neighbor_index[idx*MAX_NEIGHBOR_NUM+cur_neigh_num] = cur_nei_idx;
								neighbor_weight[idx*MAX_NEIGHBOR_NUM+cur_neigh_num] = _kernel_square(cur_dis2,radius2);
								cur_neigh_num ++;
							}
						}
					}
				}
			}
		}
		neighbor_num[idx] = cur_neigh_num;
	}
	

	__global__
	void solve_coeffs_one_kernel(const int z, const int num, const int nChannels, const float* values, const int bucket_width, const int bucket_height, const int bucket_depth,
									const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
									const int* neighbor_num, const int* neighbor_index, const float* neighbor_weight, float* coeffs)
	{
		
		for(int y = 0;y < bucket_height;y++)
		{
			for(int x = 0;x < bucket_width;x++)
			{
				int bucket_idx = z*bucket_height*bucket_width+y*bucket_width+x;
				int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
				int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];
				for(int i = 0;i < cur_bucket_stored_num;i++)
				{
					int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
					int cur_nei_num = neighbor_num[cur_coord_index];
					for(int c = 0;c < nChannels; c++)
					{
						float cur_coeff = values[cur_coord_index*nChannels+c];
						for(int j = 0;j < cur_nei_num;j++)
						{
							int cur_nei_idx = neighbor_index[cur_coord_index*MAX_NEIGHBOR_NUM+j];
							float cur_nei_weight = neighbor_weight[cur_coord_index*MAX_NEIGHBOR_NUM+j];
							cur_coeff -= coeffs[cur_nei_idx*nChannels+c]*cur_nei_weight;
						}
						coeffs[cur_coord_index*nChannels+c] = cur_coeff;
					}
				}
			}
		}
		
	}

	__global__
	void solve_coeffs_part_kernel(const int x_off, const int y_off, const int num, const int nChannels, const float* values, const int bucket_width, const int bucket_height, const int bucket_depth,							
									const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
									const int* neighbor_num, const int* neighbor_index, const float* neighbor_weight, 
									float* coeffs)
	{
		int x = blockIdx.x*2 + x_off;
		int y = blockIdx.y*2 + y_off;
		int idx = threadIdx.x;
		if(x >= bucket_width || y >= bucket_height)
			return;
			
		__shared__ int s_neighbor_index[MAX_NEIGHBOR_NUM];
		__shared__ float s_neighbor_weight[MAX_NEIGHBOR_NUM];
	
		int bucket_slice = bucket_width*bucket_height;
		
		for(int z = 0;z < bucket_depth;z++)
		{
			int bucket_idx = z*bucket_slice+y*bucket_width+x;
			int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
			int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];
			for(int i = 0;i < cur_bucket_stored_num;i++)
			{
				int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
				int cur_nei_num = neighbor_num[cur_coord_index];
				
				s_neighbor_index[idx] = neighbor_index[cur_coord_index*MAX_NEIGHBOR_NUM+idx];
				s_neighbor_weight[idx] = neighbor_weight[cur_coord_index*MAX_NEIGHBOR_NUM+idx];
				__syncthreads();
				
				if(idx < nChannels)
				{
					float cur_coeff = values[cur_coord_index*nChannels+idx];
					for(int j = 0;j < cur_nei_num;j++)
					{
						//int cur_nei_idx = neighbor_index[cur_coord_index*MAX_NEIGHBOR_NUM+j];
						//float cur_nei_weight = neighbor_weight[cur_coord_index*MAX_NEIGHBOR_NUM+j];
						
						int cur_nei_idx = s_neighbor_index[j];
						float cur_nei_weight = s_neighbor_weight[j];
						cur_coeff -= coeffs[cur_nei_idx*nChannels+idx]*cur_nei_weight;
						
					}
					coeffs[cur_coord_index*nChannels+idx] = cur_coeff;
				}
			}
		}
	}
	
	
	
	__global__
	void splat_data_part_kernel(const int x_off, const int y_off, const int z, const int bucket_width, const int bucket_height, const int bucket_depth, const int* bucket_stored_num, const int* bucket_stored_offset, 
						const int* bucket_stored_index, const float* coord_x, const float* coord_y, const float* coord_z, const float radius, const int nChannels, 
						const float* coeffs, const int out_width, const int out_height, const int out_depth, float* out_images)
	{
		int x = (blockIdx.x*blockDim.x+threadIdx.x)*2 + x_off;
		int y = (blockIdx.y*blockDim.y+threadIdx.y)*2 + y_off;
		if(x >= bucket_width || y >= bucket_height)
			return;

		int bucket_idx = z*bucket_height*bucket_width+y*bucket_width+x;
		int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
		int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];
		
		float radius2 = radius*radius;
		
		for(int i = 0;i < cur_bucket_stored_num;i++)
		{
			int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
			float cur_x = coord_x[cur_coord_index];
			float cur_y = coord_y[cur_coord_index];
			float cur_z = coord_z[cur_coord_index];
	
			for(int d = __max(0,cur_z-radius); d <= __min(out_depth-1,cur_z+radius); d++)
			{
				float radius_xy2 = radius2 - (d-cur_z)*(d-cur_z);
				float radius_xy = sqrt(radius_xy2);	
				for(int h = __max(0,cur_y-radius_xy); h <= __min(out_height-1,cur_y+radius_xy); h++)
				{
					float radius_x2 = radius_xy2 - (h-cur_y)*(h-cur_y);
					float radius_x = sqrt(radius_x2);
					for(int w = __max(0,cur_x-radius_x); w <= __min(out_width-1,cur_x+radius_x); w++)
					{
						float dis2 = (d-cur_z)*(d-cur_z)+(h-cur_y)*(h-cur_y)+(w-cur_x)*(w-cur_x);
						if(dis2 <= radius)
						{
							float cur_weight = _kernel_square(dis2,radius2);
							for(int c = 0;c < nChannels;c++)
							{
								out_images[(d*out_height*out_width+h*out_width+w)*nChannels+c] += cur_weight*coeffs[cur_coord_index*nChannels+c];
							}
						}
					}
				}
			}
		}
	}
	
	__global__
	void compute_data_kernel(const int width, const int height, const int depth, const float x_min, const float y_min, const float z_min, const float radius, 
					const int bucket_width, const int bucket_height, const int bucket_depth, const int* bucket_stored_num, const int* bucket_stored_offset, 
					const int* bucket_stored_index, const float* coord_x, const float* coord_y, const float* coord_z, const float* coeffs, const int nChannels, float* out_images)
	{
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;

		if(x >= width || y >= height)
			return;
		
		for(int z = 0;z < depth;z++)
		{
			
			int w_idx = (x - x_min)/radius;
			int h_idx = (y - y_min)/radius;
			int d_idx = (z - z_min)/radius;
			
			d_idx = __max(0,__min(bucket_depth-1,d_idx));
			h_idx = __max(0,__min(bucket_height-1,h_idx));
			w_idx = __max(0,__min(bucket_width-1,w_idx));
			
			int bucket_slice = bucket_width*bucket_height;
			
			int pixel_offset = z*height*width+y*width+x;
			float radius2 = radius*radius;
			//for(int c = 0;c < nChannels;c++)
			//	out_images[pixel_offset*nChannels+c] = 0;
				
			for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
			{ 
				for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
				{
					for(int cur_d_idx = __max(0,d_idx-1);cur_d_idx <= __min(bucket_depth-1,d_idx+1);cur_d_idx++)
					{
						int cur_bucket_idx = cur_d_idx*bucket_slice+cur_h_idx*bucket_width+cur_w_idx;
						int cur_offset = bucket_stored_offset[cur_bucket_idx];
						for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
						{
							int cur_nei_idx = bucket_stored_index[cur_offset+iii];
							
							float cur_nei_x = coord_x[cur_nei_idx];
							float cur_nei_y = coord_y[cur_nei_idx];
							float cur_nei_z = coord_z[cur_nei_idx];
							float cur_dis2 = (x-cur_nei_x)*(x-cur_nei_x)+(y-cur_nei_y)*(y-cur_nei_y)+(z-cur_nei_z)*(z-cur_nei_z);
							if(cur_dis2 <= radius2)
							{
								float cur_wei = _kernel_square(cur_dis2,radius2);
								for(int c = 0;c < nChannels;c++)
									out_images[pixel_offset*nChannels+c] += coeffs[cur_nei_idx*nChannels+c]*cur_wei;
							}
						}
					}
				}
			}
		
		}
	}


	__global__
	void Compute_coords_and_values_kernel(const int seed_width, const int seed_height, const int seed_depth, const int width, const int height, const int depth,
				const int skip, const float* u, const float* v, const float* w, const float weight1, 
				float* coord_x, float* coord_y, float* coord_z, float* values)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= seed_width || y >= seed_height)
			return ;

		for(int z = 0;z < seed_depth;z++)
		{
			int seed_offset = z*seed_height*seed_width+y*seed_width+x;
			int x_off = x*skip;
			int y_off = y*skip;
			int z_off = z*skip;
			int offset = z_off*height*width+y_off*width+x_off;

			coord_x[seed_offset] = x_off + u[offset]*(1-weight1);
			coord_y[seed_offset] = y_off + v[offset]*(1-weight1);
			coord_z[seed_offset] = z_off + w[offset]*(1-weight1);
			
			values[seed_offset*6+0] = -u[offset]*(1-weight1);
			values[seed_offset*6+1] = -v[offset]*(1-weight1);
			values[seed_offset*6+2] = -w[offset]*(1-weight1);
			values[seed_offset*6+3] = u[offset]*weight1;
			values[seed_offset*6+4] = v[offset]*weight1;
			values[seed_offset*6+5] = w[offset]*weight1;
		}
	}

	__global__
	void warp_and_blend_kernel(const int width, const int height, const int depth, const int nChannels, const float* image1, const float* image2, 
						const float* vel_image, const float weight1, float* out)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= width || y >= height)
			return ;


		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float target1_x = vel_image[offset*6+0]+x;
			float target1_y = vel_image[offset*6+1]+y;
			float target1_z = vel_image[offset*6+2]+z;
			float target2_x = vel_image[offset*6+3]+x;
			float target2_y = vel_image[offset*6+4]+y;
			float target2_z = vel_image[offset*6+5]+z;


			for(int c = 0;c < nChannels;c++)
				out[offset*nChannels+c] = 0;

			int coord_x0 = floor(target1_x);
			int coord_x1 = coord_x0+1;
			int coord_y0 = floor(target1_y);
			int coord_y1 = coord_y0+1;
			int coord_z0 = floor(target1_z);
			int coord_z1 = coord_z0+1;
			float off_x = target1_x - coord_x0;
			float off_y = target1_y - coord_y0;
			float off_z = target1_z - coord_z0;
			coord_x0 = clamp(coord_x0,0,width-1);
			coord_x1 = clamp(coord_x1,0,width-1);
			coord_y0 = clamp(coord_y0,0,height-1);
			coord_y1 = clamp(coord_y1,0,height-1);
			coord_z0 = clamp(coord_z0,0,depth-1);
			coord_z1 = clamp(coord_z1,0,depth-1);

			float input[8];
			for(int c = 0;c < nChannels;c++)
			{
				input[0] = image1[(coord_z0*height*width+coord_y0*width+coord_x0)*nChannels+c];
				input[1] = image1[(coord_z0*height*width+coord_y0*width+coord_x1)*nChannels+c];
				input[2] = image1[(coord_z0*height*width+coord_y1*width+coord_x0)*nChannels+c];
				input[3] = image1[(coord_z0*height*width+coord_y1*width+coord_x1)*nChannels+c];
				input[4] = image1[(coord_z1*height*width+coord_y0*width+coord_x0)*nChannels+c];
				input[5] = image1[(coord_z1*height*width+coord_y0*width+coord_x1)*nChannels+c];
				input[6] = image1[(coord_z1*height*width+coord_y1*width+coord_x0)*nChannels+c];
				input[7] = image1[(coord_z1*height*width+coord_y1*width+coord_x1)*nChannels+c];

				out[offset*nChannels+c] += weight1*trilinear_interpolation(input,off_x,off_y,off_z);
			}

			coord_x0 = floor(target2_x);
			coord_x1 = coord_x0+1;
			coord_y0 = floor(target2_y);
			coord_y1 = coord_y0+1;
			coord_z0 = floor(target2_z);
			coord_z1 = coord_z0+1;
			off_x = target2_x-coord_x0;
			off_y = target2_y-coord_y0;
			off_z = target2_z-coord_z0;
			coord_x0 = clamp(coord_x0,0,width-1);
			coord_x1 = clamp(coord_x1,0,width-1);
			coord_y0 = clamp(coord_y0,0,height-1);
			coord_y1 = clamp(coord_y1,0,height-1);
			coord_z0 = clamp(coord_z0,0,depth-1);
			coord_z1 = clamp(coord_z1,0,depth-1);

			
			for(int c = 0;c < nChannels;c++)
			{
				input[0] = image2[(coord_z0*height*width+coord_y0*width+coord_x0)*nChannels+c];
				input[1] = image2[(coord_z0*height*width+coord_y0*width+coord_x1)*nChannels+c];
				input[2] = image2[(coord_z0*height*width+coord_y1*width+coord_x0)*nChannels+c];
				input[3] = image2[(coord_z0*height*width+coord_y1*width+coord_x1)*nChannels+c];
				input[4] = image2[(coord_z1*height*width+coord_y0*width+coord_x0)*nChannels+c];
				input[5] = image2[(coord_z1*height*width+coord_y0*width+coord_x1)*nChannels+c];
				input[6] = image2[(coord_z1*height*width+coord_y1*width+coord_x0)*nChannels+c];
				input[7] = image2[(coord_z1*height*width+coord_y1*width+coord_x1)*nChannels+c];

				out[offset*nChannels+c] += (1-weight1)*trilinear_interpolation(input,off_x,off_y,off_z);
			}
		}
	}

	__global__
	void warp_and_blend_tricubic_kernel(const int width, const int height, const int depth, const int nChannels, const float* image1, const float* image2, 
						const float* vel_image, const float weight1, float* out)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= width || y >= height)
			return ;


		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float target1_x = vel_image[offset*6+0]+x;
			float target1_y = vel_image[offset*6+1]+y;
			float target1_z = vel_image[offset*6+2]+z;
			float target2_x = vel_image[offset*6+3]+x;
			float target2_y = vel_image[offset*6+4]+y;
			float target2_z = vel_image[offset*6+5]+z;


			for(int c = 0;c < nChannels;c++)
				out[offset*nChannels+c] = 0;

			
			float coord_x = target1_x;
			float coord_y = target1_y;
			float coord_z = target1_z;
			int ix = floor(coord_x);
			int iy = floor(coord_y);
			int iz = floor(coord_z);
			float fx = coord_x - ix;
			float fy = coord_y - iy;
			float fz = coord_z - iz;

			for(int c = 0;c < nChannels;c++)
			{
				float data_z[4];
				float dk,dk1,deltak,a3,a2,a1,a0;
				for(int k = 0;k < 4;k++)
				{
					float data_y[4];
					int tmp_z = clamp(iz-1+k,0,depth-1);
					for(int j = 0;j < 4;j++)
					{
						float data_x[4];
						int tmp_y = clamp(iy-1+j,0,height-1);
						for(int i = 0;i < 4;i++)
						{
							int tmp_x = clamp(ix-1+i,0,width-1);
							data_x[i] = image1[(tmp_z*height*width+tmp_y*width+tmp_x)*nChannels+c];
						}

						// cubic interpolation for dimension x
						dk = 0.5*(data_x[2]-data_x[0]);
						dk1 = 0.5*(data_x[3]-data_x[1]);
						deltak = data_x[2]-data_x[1];

						if(deltak == 0)
							dk = dk1 = 0;
						else
						{
							if(dk*deltak < 0)
								dk = 0;
							if(dk1*deltak < 0)
								dk1 = 0;
						}

						a3 = dk+dk1-2*deltak;
						a2 = 3*deltak - 2*dk - dk1;
						a1 = dk;
						a0 = data_x[1];

						data_y[j] = a0 + fx*(a1 + fx*(a2+fx*a3));

					}

					// cubic interpolation for dimension y
					dk = 0.5*(data_y[2]-data_y[0]);
					dk1 = 0.5*(data_y[3]-data_y[1]);
					deltak = data_y[2]-data_y[1];


					if(deltak == 0)
						dk = dk1 = 0;
					else
					{
						if(dk*deltak < 0)
							dk = 0;
						if(dk1*deltak < 0)
							dk1 = 0;
					}

					a3 = dk+dk1-2*deltak;
					a2 = 3*deltak - 2*dk - dk1;
					a1 = dk;
					a0 = data_y[1];

					data_z[k] = a0 + fy*(a1 + fy*(a2+fy*a3));
				}

				// cubic interpolation for dimension z
				dk = 0.5*(data_z[2]-data_z[0]);
				dk1 = 0.5*(data_z[3]-data_z[1]);
				deltak = data_z[2]-data_z[1];

				if(deltak == 0)
					dk = dk1 = 0;
				else
				{
					if(dk*deltak < 0)
						dk = 0;
					if(dk1*deltak < 0)
						dk1 = 0;
				}

				a3 = dk+dk1-2*deltak;
				a2 = 3*deltak - 2*dk - dk1;
				a1 = dk;
				a0 = data_z[1];
				out[offset*nChannels+c] += weight1*(a0 + fz*(a1 + fz*(a2+fz*a3)));
			}
		
			
			coord_x = target2_x;
			coord_y = target2_y;
			coord_z = target2_z;

			ix = floor(coord_x);
			iy = floor(coord_y);
			iz = floor(coord_z);
			fx = coord_x - ix;
			fy = coord_y - iy;
			fz = coord_z - iz;

			for(int c = 0;c < nChannels;c++)
			{
				float data_z[4];
				float dk,dk1,deltak,a3,a2,a1,a0;

				for(int k = 0;k < 4;k++)
				{
					float data_y[4];
					int tmp_z = clamp(iz-1+k,0,depth-1);
					for(int j = 0;j < 4;j++)
					{
						float data_x[4];
						int tmp_y = clamp(iy-1+j,0,height-1);
						for(int i = 0;i < 4;i++)
						{
							int tmp_x = clamp(ix-1+i,0,width-1);
							data_x[i] = image2[(tmp_z*height*width+tmp_y*width+tmp_x)*nChannels+c];
						}

						// cubic interpolation for dimension x
						dk = 0.5*(data_x[2]-data_x[0]);
						dk1 = 0.5*(data_x[3]-data_x[1]);
						deltak = data_x[2]-data_x[1];

						if(deltak == 0)
							dk = dk1 = 0;
						else
						{
							if(dk*deltak < 0)
								dk = 0;
							if(dk1*deltak < 0)
								dk1 = 0;
						}


						a3 = dk+dk1-2*deltak;
						a2 = 3*deltak - 2*dk - dk1;
						a1 = dk;
						a0 = data_x[1];
						data_y[j] = a0 + fx*(a1 + fx*(a2+fx*a3));
					}

					// cubic interpolation for dimension y
					dk = 0.5*(data_y[2]-data_y[0]);
					dk1 = 0.5*(data_y[3]-data_y[1]);
					deltak = data_y[2]-data_y[1];

					if(deltak == 0)
						dk = dk1 = 0;
					else
					{
						if(dk*deltak < 0)
							dk = 0;
						if(dk1*deltak < 0)
							dk1 = 0;
					}

					a3 = dk+dk1-2*deltak;
					a2 = 3*deltak - 2*dk - dk1;
					a1 = dk;
					a0 = data_y[1];

					data_z[k] = a0 + fy*(a1 + fy*(a2+fy*a3));
				}

				// cubic interpolation for dimension z
				dk = 0.5*(data_z[2]-data_z[0]);
				dk1 = 0.5*(data_z[3]-data_z[1]);
				deltak = data_z[2]-data_z[1];

				if(deltak == 0)
					dk = dk1 = 0;
				else
				{
					if(dk*deltak < 0)
						dk = 0;
					if(dk1*deltak < 0)
						dk1 = 0;
				}

				a3 = dk+dk1-2*deltak;
				a2 = 3*deltak - 2*dk - dk1;
				a1 = dk;
				a0 = data_z[1];

				out[offset*nChannels+c] += (1-weight1)*(a0 + fz*(a1 + fz*(a2+fz*a3)));
			}
		}
	}


	/***********  another implementation of neighbors begin ************/
	__global__
	void Compute_Neightbors_compute_neighbor_num_kernel(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float radius, 
					const int bucket_width, const int bucket_height, const int bucket_depth,
					const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
					int* neighbor_num)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num)
			return;

		int bucket_idx = coord_in_which_bucket[idx];
		
		int bucket_slice = bucket_width*bucket_height;
		int d_idx = bucket_idx/bucket_slice;
		int rest_idx = bucket_idx%bucket_slice;
		int w_idx = rest_idx%bucket_width;
		int h_idx = rest_idx/bucket_width;

		float cur_x = coord_x[idx];
		float cur_y = coord_y[idx];
		float cur_z = coord_z[idx];
		int cur_neigh_num = 0;
		
		float radius2 = radius*radius;
		
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{ 
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				for(int cur_d_idx = __max(0,d_idx-1);cur_d_idx <= __min(bucket_depth-1,d_idx+1);cur_d_idx++)
				{
					int cur_bucket_idx = cur_d_idx*bucket_slice+cur_h_idx*bucket_width+cur_w_idx;
					int cur_offset = bucket_stored_offset[cur_bucket_idx];
					for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
					{
						int cur_nei_idx = bucket_stored_index[cur_offset+iii];
						if(cur_nei_idx == idx)
							continue;
						float cur_nei_x = coord_x[cur_nei_idx];
						float cur_nei_y = coord_y[cur_nei_idx];
						float cur_nei_z = coord_z[cur_nei_idx];
						float cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y)+(cur_z-cur_nei_z)*(cur_z-cur_nei_z);
						if(cur_dis2 <= radius2)
						{		
							cur_neigh_num ++;
						}
					}
				}
			}
		}
		neighbor_num[idx] = cur_neigh_num;
	}
	
	__global__
	void Compute_Neightbors_fillin_neighbors_kernel(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float radius, 
					const int bucket_width, const int bucket_height, const int bucket_depth,
					const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
					const int* neighbor_offset, int* neighbor_index, float* neighbor_weight)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num)
			return;

		int bucket_idx = coord_in_which_bucket[idx];
		
		int bucket_slice = bucket_width*bucket_height;
		int d_idx = bucket_idx/bucket_slice;
		int rest_idx = bucket_idx%bucket_slice;
		int w_idx = rest_idx%bucket_width;
		int h_idx = rest_idx/bucket_width;

		float cur_x = coord_x[idx];
		float cur_y = coord_y[idx];
		float cur_z = coord_z[idx];
		int cur_neigh_num = 0;
		int cur_neigh_offset = neighbor_offset[idx];
		
		float radius2 = radius*radius;
		
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{ 
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				for(int cur_d_idx = __max(0,d_idx-1);cur_d_idx <= __min(bucket_depth-1,d_idx+1);cur_d_idx++)
				{
					int cur_bucket_idx = cur_d_idx*bucket_slice+cur_h_idx*bucket_width+cur_w_idx;
					int cur_offset = bucket_stored_offset[cur_bucket_idx];
					for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
					{
						int cur_nei_idx = bucket_stored_index[cur_offset+iii];
						if(cur_nei_idx == idx)
							continue;
						float cur_nei_x = coord_x[cur_nei_idx];
						float cur_nei_y = coord_y[cur_nei_idx];
						float cur_nei_z = coord_z[cur_nei_idx];
						float cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y)+(cur_z-cur_nei_z)*(cur_z-cur_nei_z);
						if(cur_dis2 <= radius2)
						{		
							neighbor_index[cur_neigh_offset+cur_neigh_num] = cur_nei_idx;
							neighbor_weight[cur_neigh_offset+cur_neigh_num] = _kernel_square(cur_dis2,radius2);
							cur_neigh_num ++;
						}
					}
				}
			}
		}
	}
	
	__global__
	void solve_coeffs_part_various_neighbor_num_kernel(const int x_off, const int y_off, const int num, const int nChannels, const float* values, const int bucket_width, const int bucket_height, const int bucket_depth,							
									const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
									const int* neighbor_num, const int* neighbor_offset, const int* neighbor_index, const float* neighbor_weight, 
									float* coeffs)
	{
		int x = blockIdx.x*2 + x_off;
		int y = blockIdx.y*2 + y_off;
		int idx = threadIdx.x;
		if(x >= bucket_width || y >= bucket_height || idx >= nChannels)
			return;
			
		int bucket_slice = bucket_width*bucket_height;
		
		for(int z = 0;z < bucket_depth;z++)
		{
			int bucket_idx = z*bucket_slice+y*bucket_width+x;
			int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
			int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];
			for(int i = 0;i < cur_bucket_stored_num;i++)
			{
				int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
				int cur_nei_num = neighbor_num[cur_coord_index];
				int cur_nei_offset = neighbor_offset[cur_coord_index];
				
				float cur_coeff = values[cur_coord_index*nChannels+idx];
				for(int j = 0;j < cur_nei_num;j++)
				{
					int cur_nei_idx = neighbor_index[cur_nei_offset+j];
					float cur_nei_weight = neighbor_weight[cur_nei_offset+j];
					
					cur_coeff -= coeffs[cur_nei_idx*nChannels+idx]*cur_nei_weight;
					
				}
				coeffs[cur_coord_index*nChannels+idx] = cur_coeff;
				
			}
		}
	}

	/*****************  another implementation of neighbors end ****************/

	void cu_Compute_Boundingbox(const int num, const float* coord_x, const float* coord_y, const float* coord_z, float boxmin[3], float boxmax[3])
	{
	
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,coord_x,boxmax[0]);
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,coord_y,boxmax[1]);
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,coord_z,boxmax[2]);
		ZQ_CUDA_BaseUtils::cu_Find_MIN_Value(num,coord_x,boxmin[0]);
		ZQ_CUDA_BaseUtils::cu_Find_MIN_Value(num,coord_y,boxmin[1]);
		ZQ_CUDA_BaseUtils::cu_Find_MIN_Value(num,coord_z,boxmin[2]);
	}


	
	void cu_Compute_coord_in_which_bucket(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float x_min, const float y_min, const float z_min,
			const float radius, const int bucket_width, const int bucket_height, const int bucket_depth, int* coord_in_which_bucket)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		compute_coord_in_which_bucket_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,coord_z,x_min,y_min,z_min,radius,bucket_width,bucket_height,bucket_depth,coord_in_which_bucket);
	}
	
	
	

	void cu_Compute_Neightbors(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float radius, const int bucket_width, const int bucket_height, 
									  const int bucket_depth, const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
									  int* neighbor_num, int* neighbor_index, float* neighbor_weight)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		Compute_Neightbors_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,coord_z, radius,bucket_width,bucket_height,bucket_depth,
				bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,neighbor_num,neighbor_index,neighbor_weight);
	}
	
	
	/****************    another implementation of neighbors begin **********/
	
	void cu_Compute_Neightbors_various_neighbor_num(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const float radius, const int bucket_width, const int bucket_height, 
									  const int bucket_depth, const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
									  int* neighbor_num, int* neighbor_offset, int*& neighbor_index, float*& neighbor_weight)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		
		Compute_Neightbors_compute_neighbor_num_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,coord_z, radius,bucket_width,bucket_height,bucket_depth,
				bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,neighbor_num);
		
		ZQ_CUDA_BaseUtils::cu_Compute_bucket_stored_offset(num, neighbor_num, neighbor_offset);
		
		int total_neighbor_num;
		ZQ_CUDA_BaseUtils::cu_SUM(num, neighbor_num, total_neighbor_num);
		
		checkCudaErrors( cudaMalloc((void**)&neighbor_index,sizeof(int)*total_neighbor_num) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_weight,sizeof(float)*total_neighbor_num) );
		checkCudaErrors( cudaMemset(neighbor_index,0,sizeof(int)*total_neighbor_num) );
		checkCudaErrors( cudaMemset(neighbor_weight,0,sizeof(float)*total_neighbor_num) );
		
		Compute_Neightbors_fillin_neighbors_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,coord_z, radius,bucket_width,bucket_height,bucket_depth,
				bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,neighbor_offset,neighbor_index,neighbor_weight);
	}
	
	/****************    another implementation of neighbors end **********/
	

	void cu_ScatteredInterpolation(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const int nChannels, const float* values, const float radius, 
				const int iterations,const int out_width, const int out_height, const int out_depth, float* out_images)
	{
	
		float time1 = 0, time2 = 0, time3 = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaEventRecord(start,0);
		
		dim3 bS(1,1,1);
		dim3 gS(1,1,1);
	
		float boxmin[3],boxmax[3];
		cu_Compute_Boundingbox(num,coord_x,coord_y,coord_z,boxmin,boxmax);
	
		
		int bucket_width = (boxmax[0] - boxmin[0])/radius + 1;
		int bucket_height = (boxmax[1] - boxmin[1])/radius + 1;
		int bucket_depth = (boxmax[2] - boxmin[2])/radius + 1;
		int bucket_num = bucket_width*bucket_height*bucket_depth;
		int* bucket_stored_num = 0;
		int* bucket_stored_offset = 0;
		int* bucket_stored_index = 0;
		int* coord_in_which_bucket = 0;
		checkCudaErrors( cudaMalloc((void**)&bucket_stored_num,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMalloc((void**)&bucket_stored_offset,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMalloc((void**)&bucket_stored_index,sizeof(int)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_in_which_bucket,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(bucket_stored_num,0,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMemset(bucket_stored_offset,0,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMemset(bucket_stored_index,0,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(coord_in_which_bucket,0,sizeof(int)*num) );
		
		cu_Compute_coord_in_which_bucket(num,coord_x,coord_y,coord_z,boxmin[0],boxmin[1],boxmin[2],radius,bucket_width,bucket_height,bucket_depth,coord_in_which_bucket);
		
		ZQ_CUDA_BaseUtils::cu_Distribute_Bucket(num,bucket_num,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket);
					

		/*int* sto_num = new int[bucket_num];
		checkCudaErrors( cudaMemcpy(sto_num,bucket_stored_num,sizeof(int)*bucket_num,cudaMemcpyDeviceToHost) );
		FILE* out_sto = fopen("sto.txt","w");
		for(int i = 0;i < bucket_num;i++)
		{
			fprintf(out_sto,"%d\n",sto_num[i]);;
		}
		fclose(out_sto);
		delete []sto_num;*/
		
		/*int* coord_which = new int[num];
		checkCudaErrors( cudaMemcpy(coord_which,coord_in_which_bucket,sizeof(int)*num,cudaMemcpyDeviceToHost) );
		FILE* out_which = fopen("which.txt","w");
		for(int i = 0;i < num;i++)
		{
			fprintf(out_which,"%d\n",coord_which[i]);;
		}
		fclose(out_which);
		delete []coord_which;*/
		
		
		int* neighbor_num = 0;
		int* neighbor_index = 0;
		float* neighbor_weight = 0;
		checkCudaErrors( cudaMalloc((void**)&neighbor_num,sizeof(int)*num) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_index,sizeof(int)*num*MAX_NEIGHBOR_NUM) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_weight,sizeof(float)*num*MAX_NEIGHBOR_NUM) );
		checkCudaErrors( cudaMemset(neighbor_num,0,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(neighbor_index,0,sizeof(int)*num*MAX_NEIGHBOR_NUM) );
		checkCudaErrors( cudaMemset(neighbor_weight,0,sizeof(float)*num*MAX_NEIGHBOR_NUM) );
		cu_Compute_Neightbors(num,coord_x,coord_y,coord_z,radius,bucket_width,bucket_height,bucket_depth,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,
			neighbor_num,neighbor_index,neighbor_weight);
	
		
		int max_n_num = 0;
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,neighbor_num,max_n_num);
		printf("max_neighbor/hardcode_allocation=%d/%d\n",max_n_num,MAX_NEIGHBOR_NUM);		

		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time1,start,stop);
		
		cudaEventRecord(start,0);
		
		float* coeffs = 0;
		checkCudaErrors( cudaMalloc((void**)&coeffs,sizeof(float)*num*nChannels) );
		checkCudaErrors( cudaMemset(coeffs,0,sizeof(float)*num*nChannels) );
		
		
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize(((bucket_width+1)/2+blockSize.x-1)/blockSize.x,((bucket_height+1)/2+blockSize.y-1)/blockSize.y);
		
		
		

		dim3 blockSize1(MAX_NEIGHBOR_NUM,1);
		dim3 gridSize1((bucket_width+1)/2,(bucket_height+1)/2);
		
		for(int it = 0;it < iterations;it++)
		{
			int x_off[4] = {0,0,1,1};
			int y_off[4] = {0,1,0,1};

			for(int iii = 0;iii < 4;iii++)
			{
				solve_coeffs_part_kernel<<<gridSize1,blockSize1>>>(x_off[iii],y_off[iii],num,nChannels,values,bucket_width,bucket_height,bucket_depth,
					bucket_stored_num,bucket_stored_offset,bucket_stored_index,neighbor_num,neighbor_index,neighbor_weight,
					coeffs);
			}		
		}

		checkCudaErrors( cudaFree(neighbor_num) );
		checkCudaErrors( cudaFree(neighbor_index) );
		checkCudaErrors( cudaFree(neighbor_weight) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time2,start,stop);
		cudaEventRecord(start,0);
		
		if(0)
		{
			dim3 blockSize2(2,1);
			int gridSize2_width = ((bucket_width+1)/2+blockSize2.x-1)/blockSize2.x;
			int gridSize2_height = ((bucket_height+1)/2+blockSize2.y-1)/blockSize2.y;
			dim3 gridSize2(gridSize2_width,gridSize2_height);
			for(int z = 0;z < bucket_depth;z++)
			{
				int x_off[4] = {0,0,1,1};
				int y_off[4] = {0,1,0,1};
				for(int iii = 0;iii < 4;iii++)
				{
					splat_data_part_kernel<<<gridSize2,blockSize2>>>(x_off[iii],y_off[iii],z,bucket_width,bucket_height,bucket_depth,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
						coord_x,coord_y,coord_z,radius,nChannels,coeffs,out_width,out_height,out_depth,out_images);
				}
			}
		}
		else
		{
			dim3 blockSize2(BLOCK_SIZE,BLOCK_SIZE,1);
			int gridSize2_width = (out_width+blockSize2.x-1)/blockSize2.x;
			int gridSize2_height = (out_height+blockSize2.y-1)/blockSize2.y;	
			dim3 gridSize2(gridSize2_width,gridSize2_height,1);
			compute_data_kernel<<<gridSize2,blockSize2>>>(out_width,out_height,out_depth,boxmin[0],boxmin[1],boxmin[2],radius,bucket_width,bucket_height,bucket_depth,bucket_stored_num,
				bucket_stored_offset,bucket_stored_index,coord_x,coord_y,coord_z,coeffs,nChannels,out_images);
			
		}
		
		checkCudaErrors( cudaFree(coeffs) );
		checkCudaErrors( cudaFree(bucket_stored_num) );
		checkCudaErrors( cudaFree(bucket_stored_offset) );
		checkCudaErrors( cudaFree(bucket_stored_index) );
		checkCudaErrors( cudaFree(coord_in_which_bucket) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time3,start,stop);
		
		printf("prepare=%f,solve=%f,splat=%f\n",time1*0.001,time2*0.001,time3*0.001);
	}
	
	void cu_ScatteredInterpolation_various_neighbor_num(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const int nChannels, const float* values, const float radius, 
				const int iterations,const int out_width, const int out_height, const int out_depth, float* out_images)
	{
	
		float time1 = 0, time2 = 0, time3 = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaEventRecord(start,0);
		
		dim3 bS(1,1,1);
		dim3 gS(1,1,1);
	
		float boxmin[3],boxmax[3];
		cu_Compute_Boundingbox(num,coord_x,coord_y,coord_z,boxmin,boxmax);
	
		
		int bucket_width = (boxmax[0] - boxmin[0])/radius + 1;
		int bucket_height = (boxmax[1] - boxmin[1])/radius + 1;
		int bucket_depth = (boxmax[2] - boxmin[2])/radius + 1;
		int bucket_num = bucket_width*bucket_height*bucket_depth;
		int* bucket_stored_num = 0;
		int* bucket_stored_offset = 0;
		int* bucket_stored_index = 0;
		int* coord_in_which_bucket = 0;
		checkCudaErrors( cudaMalloc((void**)&bucket_stored_num,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMalloc((void**)&bucket_stored_offset,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMalloc((void**)&bucket_stored_index,sizeof(int)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_in_which_bucket,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(bucket_stored_num,0,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMemset(bucket_stored_offset,0,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMemset(bucket_stored_index,0,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(coord_in_which_bucket,0,sizeof(int)*num) );
		
		cu_Compute_coord_in_which_bucket(num,coord_x,coord_y,coord_z,boxmin[0],boxmin[1],boxmin[2],radius,bucket_width,bucket_height,bucket_depth,coord_in_which_bucket);
		
		
		
		ZQ_CUDA_BaseUtils::cu_Distribute_Bucket(num,bucket_num,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket);
					
		
		int* neighbor_num = 0;
		int* neighbor_offset = 0;
		int* neighbor_index = 0;
		float* neighbor_weight = 0;
		checkCudaErrors( cudaMalloc((void**)&neighbor_num,sizeof(int)*num) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_offset,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(neighbor_num,0,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(neighbor_offset,0,sizeof(int)*num) );
	
		cu_Compute_Neightbors_various_neighbor_num(num,coord_x,coord_y,coord_z,radius,bucket_width,bucket_height,bucket_depth,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,
			neighbor_num,neighbor_offset,neighbor_index,neighbor_weight);
	
		
		int max_n_num = 0;
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,neighbor_num,max_n_num);
		printf("max_neighbor/hardcode_allocation=%d/%d\n",max_n_num,MAX_NEIGHBOR_NUM);		

		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time1,start,stop);
		
		cudaEventRecord(start,0);
		
		float* coeffs = 0;
		checkCudaErrors( cudaMalloc((void**)&coeffs,sizeof(float)*num*nChannels) );
		checkCudaErrors( cudaMemset(coeffs,0,sizeof(float)*num*nChannels) );
		
		
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize(((bucket_width+1)/2+blockSize.x-1)/blockSize.x,((bucket_height+1)/2+blockSize.y-1)/blockSize.y);
		
		
		

		dim3 blockSize1(nChannels,1);
		dim3 gridSize1((bucket_width+1)/2,(bucket_height+1)/2);
		
		for(int it = 0;it < iterations;it++)
		{
			int x_off[4] = {0,0,1,1};
			int y_off[4] = {0,1,0,1};

			for(int iii = 0;iii < 4;iii++)
			{
				solve_coeffs_part_various_neighbor_num_kernel<<<gridSize1,blockSize1>>>(x_off[iii],y_off[iii],num,nChannels,values,bucket_width,bucket_height,bucket_depth,
					bucket_stored_num,bucket_stored_offset,bucket_stored_index,neighbor_num,neighbor_offset,neighbor_index,neighbor_weight,
					coeffs);
			}		
		}

		checkCudaErrors( cudaFree(neighbor_num) );
		checkCudaErrors( cudaFree(neighbor_offset) );
		checkCudaErrors( cudaFree(neighbor_index) );
		checkCudaErrors( cudaFree(neighbor_weight) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time2,start,stop);
		cudaEventRecord(start,0);
		
		
		dim3 blockSize2(BLOCK_SIZE,BLOCK_SIZE,1);
		int gridSize2_width = (out_width+blockSize2.x-1)/blockSize2.x;
		int gridSize2_height = (out_height+blockSize2.y-1)/blockSize2.y;	
		dim3 gridSize2(gridSize2_width,gridSize2_height,1);
		compute_data_kernel<<<gridSize2,blockSize2>>>(out_width,out_height,out_depth,boxmin[0],boxmin[1],boxmin[2],radius,bucket_width,bucket_height,bucket_depth,bucket_stored_num,
			bucket_stored_offset,bucket_stored_index,coord_x,coord_y,coord_z,coeffs,nChannels,out_images);
			
		
		
		checkCudaErrors( cudaFree(coeffs) );
		checkCudaErrors( cudaFree(bucket_stored_num) );
		checkCudaErrors( cudaFree(bucket_stored_offset) );
		checkCudaErrors( cudaFree(bucket_stored_index) );
		checkCudaErrors( cudaFree(coord_in_which_bucket) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time3,start,stop);
		
		printf("prepare=%f,solve=%f,splat=%f\n",time1*0.001,time2*0.001,time3*0.001);
	}


	void cu_BlendTwoImages(const int width, const int height, const int depth, const int nChannels, const float* image1, const float* image2, 
				const float* u, const float* v, const float* w, const float weight1, 
				const int skip, const float radius, const int iterations, float* out_image, bool various_neighbor_num, bool cubic)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		float* coord_x = 0;
		float* coord_y = 0;
		float* coord_z = 0;
		float* values = 0;
		int seed_width = width/skip;
		int seed_height = height/skip;
		int seed_depth = depth/skip;
		int num = seed_width*seed_height*seed_depth;
		checkCudaErrors( cudaMalloc((void**)&coord_x,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_y,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_z,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&values,sizeof(float)*num*6) );
		checkCudaErrors( cudaMemset(coord_x,0,sizeof(float)*num) );
		checkCudaErrors( cudaMemset(coord_y,0,sizeof(float)*num) );
		checkCudaErrors( cudaMemset(coord_z,0,sizeof(float)*num) );
		checkCudaErrors( cudaMemset(values,0,sizeof(float)*num*6) );
		
		float time1 = 0;
		cudaEvent_t start1,stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1,0);
		
		Compute_coords_and_values_kernel<<<gridSize,blockSize>>>(seed_width,seed_height,seed_depth,width,height,depth,skip,u,v,w,weight1,coord_x,coord_y,coord_z,values);
		
		cudaEventRecord(stop1,0);
		cudaEventSynchronize(start1);
		cudaEventSynchronize(stop1);
		cudaEventElapsedTime(&time1,start1,stop1);
		
		printf("compute coords:%f\n",0.001*time1);
		
		float* vel_image = 0;
		checkCudaErrors( cudaMalloc((void**)&vel_image,sizeof(float)*width*height*depth*6) );
		checkCudaErrors( cudaMemset(vel_image,0,sizeof(float)*width*height*depth*6) );
	
		if(various_neighbor_num)
			cu_ScatteredInterpolation_various_neighbor_num(num, coord_x, coord_y, coord_z, 6, values, radius, iterations, width, height, depth, vel_image);
		else
			cu_ScatteredInterpolation(num, coord_x, coord_y, coord_z, 6, values, radius, iterations, width, height, depth, vel_image);
		
		float time2 = 0;
		cudaEvent_t start2,stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2,0);
		
		if(!cubic)
			warp_and_blend_kernel<<<gridSize,blockSize>>>(width, height, depth, nChannels, image1, image2, vel_image, weight1, out_image);
		else
			warp_and_blend_tricubic_kernel<<<gridSize,blockSize>>>(width, height, depth, nChannels, image1, image2, vel_image, weight1, out_image);
			
		cudaEventRecord(stop2,0);
		cudaEventSynchronize(start2);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&time2,start2,stop2);
		
		printf("warp_and_blend:%f\n",0.001*time2);
		
		checkCudaErrors( cudaFree(coord_x) );
		checkCudaErrors( cudaFree(coord_y) );
		checkCudaErrors( cudaFree(coord_z) );
		checkCudaErrors( cudaFree(values) );
		checkCudaErrors( cudaFree(vel_image) );
	}

	extern "C"
	float Cutil_ScatteredInterpolation3D(const int num, const float* coord_x, const float* coord_y, const float* coord_z, const int nChannels, const float* values, const float radius, const int iterations,
								   const int out_width, const int out_height, const int out_depth, float* out_images, bool various_neighbor_num)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* coord_x_d = 0;
		float* coord_y_d = 0;
		float* coord_z_d = 0;
		float* values_d = 0; 
		checkCudaErrors( cudaMalloc((void**)&coord_x_d,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_y_d,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_z_d,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&values_d,sizeof(float)*num*nChannels) );
		checkCudaErrors( cudaMemcpy(coord_x_d,coord_x,sizeof(float)*num,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(coord_y_d,coord_y,sizeof(float)*num,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(coord_z_d,coord_z,sizeof(float)*num,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(values_d,values,sizeof(float)*num*nChannels,cudaMemcpyHostToDevice) );

		float* out_images_d = 0;
		checkCudaErrors( cudaMalloc((void**)&out_images_d,sizeof(float)*out_width*out_height*nChannels) );
		checkCudaErrors( cudaMemset(out_images_d,0,sizeof(float)*out_width*out_height*nChannels) );

		if(various_neighbor_num)
			cu_ScatteredInterpolation_various_neighbor_num(num,coord_x_d,coord_y_d,coord_z_d,nChannels,values_d,radius,iterations,out_width,out_height,out_depth,out_images_d);
		else
			cu_ScatteredInterpolation(num,coord_x_d,coord_y_d,coord_z_d,nChannels,values_d,radius,iterations,out_width,out_height,out_depth,out_images_d);
		
		checkCudaErrors( cudaMemcpy(out_images,out_images_d,sizeof(float)*out_width*out_height*out_depth*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(coord_x_d) );
		checkCudaErrors( cudaFree(coord_y_d) );
		checkCudaErrors( cudaFree(coord_z_d) );
		checkCudaErrors( cudaFree(values_d) );
		checkCudaErrors( cudaFree(out_images_d) );

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float Cutil_BlendTwoImages3D(const int width, const int height, const int depth, const int nChannels, const float* image1, const float* image2,
			const float* u, const float* v, const float* w, const float weight1, 
			const int skip, const float radius, const int iterations, float* out_image, bool various_neighbor_num, bool cubic)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* image1_d = 0;
		float* image2_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		float* out_image_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&image1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&image2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&out_image_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(image1_d,image1,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(image2_d,image2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(out_image_d,0,sizeof(float)*width*height*depth*nChannels) );
		
		cu_BlendTwoImages(width, height, depth, nChannels, image1_d, image2_d, u_d, v_d, w_d, weight1, skip, radius, iterations, out_image_d,various_neighbor_num,cubic);
		
		checkCudaErrors( cudaMemcpy(out_image,out_image_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(image1_d) );
		checkCudaErrors( cudaFree(image2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(out_image_d) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

}

#endif