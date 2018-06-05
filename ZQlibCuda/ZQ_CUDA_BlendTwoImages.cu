#ifndef _ZQ_CUDA_BLEND_TWO_IMAGES_CU_
#define _ZQ_CUDA_BLEND_TWO_IMAGES_CU_


#include "ZQ_CUDA_BaseUtils.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"
#include "ZQ_CUDA_BlendTwoImages.cuh"

#define MAX_NEIGHBOR_NUM 32

#define _kernel_square wendland_square

namespace ZQ_CUDA_BlendTwoImages
{
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

	__device__  /*input = {I(y0,x0),I(y0,x1),I(y1,x0),I(y1,x1)}, x \in [0,1], y \in [0,1]*/
	float bilinear_interpolation(const float* input, const float x, const float y)
	{
		float w00 = (1-y)*(1-x);
		float w01 = (1-y)*x;
		float w10 = y*(1-x);
		float w11 = y*x;

		return input[0]*w00 + input[1]*w01 + input[2]*w10 + input[3]*w11;
	}

	__global__
	void compute_coord_in_which_bucket_kernel(const int num, const float* coord_x, const float* coord_y, const float x_min, const float y_min,
								const float radius, const int bucket_width, const int bucket_height, int* coord_in_which_bucket)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx >= num)
			return ;
			
		int h_idx = (coord_y[idx] - y_min)/radius;
		int w_idx = (coord_x[idx] - x_min)/radius;
			
		h_idx = __max(0,__min(bucket_height-1,h_idx));
		w_idx = __max(0,__min(bucket_width-1,w_idx));
		int bucket_idx = h_idx*bucket_width+w_idx;
		coord_in_which_bucket[idx] = bucket_idx;
	}

	

	__global__
	void Compute_Neightbors_kernel(const int num, const float* coord_x, const float* coord_y, const float radius, const int bucket_width, const int bucket_height, 
								   const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
								   int* neighbor_num, int* neighbor_index, float* neighbor_weight)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num)
			return;

		float radius2 = radius*radius;
		int bucket_idx = coord_in_which_bucket[idx];
		int w_idx = bucket_idx%bucket_width;
		int h_idx = bucket_idx/bucket_width;

		float cur_x = coord_x[idx];
		float cur_y = coord_y[idx];
		int cur_neigh_num = 0;
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				int cur_bucket_idx = cur_h_idx*bucket_width+cur_w_idx;
				int cur_offset = bucket_stored_offset[cur_bucket_idx];
				for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
				{
					int cur_nei_idx = bucket_stored_index[cur_offset+iii];
					if(cur_nei_idx == idx)
						continue;
					float cur_nei_x = coord_x[cur_nei_idx];
					float cur_nei_y = coord_y[cur_nei_idx];
					float cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y);
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
		neighbor_num[idx] = cur_neigh_num;
		/*float sum_weight = 0;
		for(int i = 0;i < cur_neigh_num;i++)
			sum_weight += neighbor_weight[idx*MAX_NEIGHBOR_NUM+i];
		if(sum_weight != 0)
		{
			for(int i = 0;i < cur_neigh_num;i++)
				neighbor_weight[idx*MAX_NEIGHBOR_NUM+i] /= sum_weight;
		}*/	
	}

	__global__
	void solve_coeffs_one_kernel(const int num, const int nChannels, const float* values, const int bucket_width, const int bucket_height,
									const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
									const int* neighbor_num, const int* neighbor_index, const float* neighbor_weight, float* coeffs)
	{
		//int x = blockIdx.x*blockDim.x+threadIdx.x;
		//int y = blockIdx.y*blockDim.y+threadIdx.y;
		//if(x != 0|| y != 0)
		//	return;
			
		for(int idx = 0;idx < num;idx++)
		{
			int nei_num = neighbor_num[idx];
			for(int c = 0;c < nChannels;c++)
			{
				float cur_coeff = values[idx*nChannels+c];
				for(int j = 0;j < nei_num;j++)
				{	
					int nei_idx = neighbor_index[idx*MAX_NEIGHBOR_NUM+j];
					float nei_wei = neighbor_weight[idx*MAX_NEIGHBOR_NUM+j];
					cur_coeff -= coeffs[nei_idx*nChannels+c]*nei_wei;
				}
				coeffs[idx*nChannels+c] = cur_coeff;
			}
		}
	}

	__global__
	void solve_coeffs_part_kernel(const int x_off, const int y_off, const int num, const int nChannels, const float* values, const int bucket_width, const int bucket_height,
									const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
									const int* neighbor_num, const int* neighbor_index, const float* neighbor_weight, float* coeffs)
	{
		int x = (blockIdx.x*blockDim.x+threadIdx.x)*2 + x_off;
		int y = (blockIdx.y*blockDim.y+threadIdx.y)*2 + y_off;
		if(x >= bucket_width || y >= bucket_height)
			return;

		int bucket_idx = y*bucket_width+x;
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

	

	__global__
	void splat_data_one_kernel(const int num, const float* coord_x, const float* coord_y, const float radius, const int nChannels, const float* coeffs, const int out_width, const int out_height, float* out_images, float* out_weight)
	{
		//int x = blockIdx.x*blockDim.x+threadIdx.x;
		//int y = blockIdx.y*blockDim.y+threadIdx.y;
		//if(x != 0 || y != 0)
		//	return;
		
		float radius2 = radius*radius;
		
		for(int idx = 0; idx < num;idx++)
		{
			float cur_x = coord_x[idx];
			float cur_y = coord_y[idx];
			
			//int h = __min(out_height-1,__max(0,cur_y));
			//int w = __min(out_width-1,__max(0,cur_x));	
			//for(int c = 0;c < nChannels;c++)
			//	out_images[(h*out_width+w)*nChannels+c] = coeffs[idx*nChannels+c];

			for(int h = __max(0,cur_y-radius); h <= __min(out_height-1,cur_y+radius); h++)
			{
				float radius_x2 = radius2 - (h-cur_y)*(h-cur_y);
				float radius_x = sqrt(radius_x2);
				for(int w = __max(0,cur_x-radius_x); w <= __min(out_width-1,cur_x+radius_x); w++)
				{
					float dis2 = (h-cur_y)*(h-cur_y)+(w-cur_x)*(w-cur_x);
					if(dis2 <= radius2)
					{
						float cur_weight = _kernel_square(dis2,radius2);
						out_weight[h*out_width+w] += cur_weight;
						for(int c = 0;c < nChannels;c++)
						{
							out_images[(h*out_width+w)*nChannels+c] += cur_weight*coeffs[idx*nChannels+c];
						}
					}
				}
			}
		}
		
	}
	
	__global__
	void splat_data_part_kernel(const int x_off, const int y_off, const int bucket_width, const int bucket_height, const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
								  const float* coord_x, const float* coord_y, const float radius, const int nChannels, const float* coeffs, const int out_width, const int out_height, float* out_images, float* out_weight)
	{
		int x = (blockIdx.x*blockDim.x+threadIdx.x)*2 + x_off;
		int y = (blockIdx.y*blockDim.y+threadIdx.y)*2 + y_off;
		if(x >= bucket_width || y >= bucket_height)
			return;
			
		float radius2 = radius*radius;

		int bucket_idx = y*bucket_width+x;
		int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
		int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];
		for(int i = 0;i < cur_bucket_stored_num;i++)
		{
			int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
			float cur_x = coord_x[cur_coord_index];
			float cur_y = coord_y[cur_coord_index];

			for(int h = __max(0,cur_y-radius); h <= __min(out_height-1,cur_y+radius); h++)
			{
				float radius_x2 = radius2 - (h-cur_y)*(h-cur_y);
				float radius_x = sqrt(radius_x2);
				for(int w = __max(0,cur_x-radius_x); w <= __min(out_width-1,cur_x+radius_x); w++)
				{
					float dis2 = (h-cur_y)*(h-cur_y)+(w-cur_x)*(w-cur_x);
					if(dis2 <= radius2)
					{
						float cur_weight = _kernel_square(dis2,radius2);
						out_weight[h*out_width+w] += cur_weight;
						for(int c = 0;c < nChannels;c++)
						{
							out_images[(h*out_width+w)*nChannels+c] += cur_weight*coeffs[cur_coord_index*nChannels+c];
						}
					}
				}
			}
		}

	}

	__global__
	void forward_move_velocity_kernel(const int width, const int height, const float weight1, const float* u, const float* v, float* tmp_vel_image, bool* keep_mask)
	{
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		float fw_u = (1.0f - weight1)*u[offset];
		float fw_v = (1.0f - weight1)*v[offset];
		float bw_u = weight1*u[offset];
		float bw_v = weight1*v[offset];
		int coord_x = x + fw_u + 0.5f;
		int coord_y = y + fw_v + 0.5f;
		if (coord_x >= 0 && coord_x <= width - 1 && coord_y >= 0 && coord_y <= height - 1)
		{
			int cur_offset = coord_y*width + coord_x;
			keep_mask[cur_offset] = true;
			tmp_vel_image[cur_offset * 4 + 0] = -fw_u;
			tmp_vel_image[cur_offset * 4 + 1] = -fw_v;
			tmp_vel_image[cur_offset * 4 + 2] = bw_u;
			tmp_vel_image[cur_offset * 4 + 3] = bw_v;
		}
	}
	__global__
	void velocity_addwith_subscript_kernel(float* vel_image, const int width, const int height)
	{
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		
		vel_image[offset * 4 + 0] += x;
		vel_image[offset * 4 + 1] += y;
		vel_image[offset * 4 + 2] += x;
		vel_image[offset * 4 + 3] += y;
	}

	__global__
	void compute_data_kernel(const int width, const int height,const float x_min, const float y_min, const float radius, 
					const int bucket_width, const int bucket_height, const int* bucket_stored_num, const int* bucket_stored_offset, 
					const int* bucket_stored_index, const float* coord_x, const float* coord_y, const float* coeffs, const int nChannels, float* out_images)
	{
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;

		if(x >= width || y >= height)
			return;

		int w_idx = (x - x_min)/radius;
		int h_idx = (y - y_min)/radius;
		
		h_idx = __max(0,__min(bucket_height-1,h_idx));
		w_idx = __max(0,__min(bucket_width-1,w_idx));
		
		int pixel_offset = y*width+x;
		float radius2 = radius*radius;
		//for(int c = 0;c < nChannels;c++)
		//	out_images[pixel_offset*nChannels+c] = 0;
			
		float sum_weight = 0;
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{ 
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				int cur_bucket_idx = cur_h_idx*bucket_width+cur_w_idx;
				int cur_offset = bucket_stored_offset[cur_bucket_idx];
				for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
				{
					int cur_nei_idx = bucket_stored_index[cur_offset+iii];
						
					float cur_nei_x = coord_x[cur_nei_idx];
					float cur_nei_y = coord_y[cur_nei_idx];
				
					float cur_dis2 = (x-cur_nei_x)*(x-cur_nei_x)+(y-cur_nei_y)*(y-cur_nei_y);
					if(cur_dis2 <= radius2)
					{
						float cur_wei = _kernel_square(cur_dis2,radius2);
						sum_weight += cur_wei;
						for(int c = 0;c < nChannels;c++)
							out_images[pixel_offset*nChannels+c] += coeffs[cur_nei_idx*nChannels+c]*cur_wei;
					}
					
				}
			}
		}
		/*if(sum_weight != 0)
		{	
			for(int c = 0;c < nChannels;c++)
				out_images[pixel_offset*nChannels+c] /= sum_weight;
		}*/
	}
	

	__global__
	void Compute_coords_and_values_kernel(const int seed_width, const int seed_height, const int width, const int height, const int skip, const float* u, const float* v, const float weight1, 
			float* coord_x, float* coord_y, float* values)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= seed_width || y >= seed_height)
			return ;

		int seed_offset = y*seed_width+x;
		int x_off = x*skip;
		int y_off = y*skip;
		int offset = y_off*width+x_off;

		coord_x[seed_offset] = x_off + u[offset]*(1-weight1);
		coord_y[seed_offset] = y_off + v[offset]*(1-weight1);
		
		values[seed_offset*4+0] = -u[offset]*(1-weight1);
		values[seed_offset*4+1] = -v[offset]*(1-weight1);
		values[seed_offset*4+2] = u[offset]*weight1;
		values[seed_offset*4+3] = v[offset]*weight1;
		
		/*values[seed_offset*4+0] = x_off;
		values[seed_offset*4+1] = y_off;
		values[seed_offset*4+2] = x_off+u[offset];
		values[seed_offset*4+3] = y_off+v[offset];*/
	}

	__global__
	void warp_and_blend_kernel(const int width, const int height, const int nChannels, const float* image1, const float* image2, 
						const float* vel_image, const float weight1, float* out, const int blend_mode)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;

		float target1_x = vel_image[offset*4+0]+x;
		float target1_y = vel_image[offset*4+1]+y;
		float target2_x = vel_image[offset*4+2]+x;
		float target2_y = vel_image[offset*4+3]+y;
		
		float real_weight1 = 0;
		float real_weight2 = 0;
		if (blend_mode == 1)
		{
			real_weight1 = 1.0f;
			real_weight2 = 0.0f;
		}
		else if (blend_mode == 2)
		{
			real_weight1 = 0.0f;
			real_weight2 = 1.0f;
		}
		else
		{
			real_weight1 = weight1;
			real_weight2 = 1.0f - weight1;
		}
			
		for(int c = 0;c < nChannels;c++)
			out[offset*nChannels+c] = 0;

		int coord_x0 = floor(target1_x);
		int coord_x1 = coord_x0+1;
		int coord_y0 = floor(target1_y);
		int coord_y1 = coord_y0+1;
		float off_x = target1_x - coord_x0;
		float off_y = target1_y - coord_y0;
		coord_x0 = clamp(coord_x0,0,width-1);
		coord_x1 = clamp(coord_x1,0,width-1);
		coord_y0 = clamp(coord_y0,0,height-1);
		coord_y1 = clamp(coord_y1,0,height-1);

		float input[4];
		for(int c = 0;c < nChannels;c++)
		{
			input[0] = image1[(coord_y0*width+coord_x0)*nChannels+c];
			input[1] = image1[(coord_y0*width+coord_x1)*nChannels+c];
			input[2] = image1[(coord_y1*width+coord_x0)*nChannels+c];
			input[3] = image1[(coord_y1*width+coord_x1)*nChannels+c];

			out[offset*nChannels+c] += real_weight1*bilinear_interpolation(input,off_x,off_y);
		}

		coord_x0 = floor(target2_x);
		coord_x1 = coord_x0+1;
		coord_y0 = floor(target2_y);
		coord_y1 = coord_y0+1;
		off_x = target2_x-coord_x0;
		off_y = target2_y-coord_y0;
		coord_x0 = clamp(coord_x0,0,width-1);
		coord_x1 = clamp(coord_x1,0,width-1);
		coord_y0 = clamp(coord_y0,0,height-1);
		coord_y1 = clamp(coord_y1,0,height-1);

		for(int c = 0;c < nChannels;c++)
		{
			input[0] = image2[(coord_y0*width+coord_x0)*nChannels+c];
			input[1] = image2[(coord_y0*width+coord_x1)*nChannels+c];
			input[2] = image2[(coord_y1*width+coord_x0)*nChannels+c];
			input[3] = image2[(coord_y1*width+coord_x1)*nChannels+c];

			out[offset*nChannels+c] += real_weight2*bilinear_interpolation(input,off_x,off_y);
		}
	}
	

	__global__
	void warp_and_blend_bicubic_kernel(const int width, const int height, const int nChannels, const float* image1, const float* image2, 
						const float* vel_image, const float weight1, float* out, const int blend_mode)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;

		float target1_x = vel_image[offset*4+0]+x;
		float target1_y = vel_image[offset*4+1]+y;
		float target2_x = vel_image[offset*4+2]+x;
		float target2_y = vel_image[offset*4+3]+y;
		
		float real_weight1 = 0;
		float real_weight2 = 0;
		if (blend_mode == 1)
		{
			real_weight1 = 1.0f;
			real_weight2 = 0.0f;
		}
		else if (blend_mode == 2)
		{
			real_weight1 = 0.0f;
			real_weight2 = 1.0f;
		}
		else
		{
			real_weight1 = weight1;
			real_weight2 = 1.0f - weight1;
		}

		for(int c = 0;c < nChannels;c++)
			out[offset*nChannels+c] = 0;
			
			
		float coord_x = target1_x;
		float coord_y = target1_y;

		int ix = floor(coord_x);
		int iy = floor(coord_y);
		float fx = coord_x - ix;
		float fy = coord_y - iy;

		for(int c = 0;c < nChannels;c++)
		{
			float data_y[4];
			float dk,dk1,deltak,a3,a2,a1,a0;
			for(int i = 0;i < 4;i++)
			{
				float data_x[4];
				int tmp_y = clamp(iy-1+i,0,height-1);
				for(int j = 0;j < 4;j++)
				{
					int tmp_x = clamp(ix-1+j,0,width-1);
					data_x[j] = image1[(tmp_y*width+tmp_x)*nChannels+c];
				}

				// bicubic interpolation for dimension y
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

				data_y[i] = a0 + fx*(a1 + fx*(a2+fx*a3));

			}

			// bicubic interpolation for dimension x
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

			out[offset*nChannels+c] += real_weight1*(a0 + fy*(a1 + fy*(a2+fy*a3)));
		}

		coord_x = target2_x;
		coord_y = target2_y;

		ix = floor(coord_x);
		iy = floor(coord_y);
		fx = coord_x - ix;
		fy = coord_y - iy;

		for(int c = 0;c < nChannels;c++)
		{
			float data_y[4];
			float dk,dk1,deltak,a3,a2,a1,a0;
			for(int i = 0;i < 4;i++)
			{
				float data_x[4];
				int tmp_y = clamp(iy-1+i,0,height-1);
				for(int j = 0;j < 4;j++)
				{
					int tmp_x = clamp(ix-1+j,0,width-1);
					data_x[j] = image2[(tmp_y*width+tmp_x)*nChannels+c];
				}

				// bicubic interpolation for dimension y
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

				data_y[i] = a0 + fx*(a1 + fx*(a2+fx*a3));

			}

			// bicubic interpolation for dimension x
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

			out[offset*nChannels+c] += real_weight2*(a0 + fy*(a1 + fy*(a2+fy*a3)));
		}

	}

	__global__ void warp_and_blend_interger_kernel(const int width, const int height, const int nChannels, const float* image1, const float* image2,
		const float* vel_image, const float weight1, float* out, const int blend_mode)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;
		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		int target1_x = vel_image[offset * 4 + 0] + x + 0.5f;
		int target1_y = vel_image[offset * 4 + 1] + y + 0.5f;
		int target2_x = vel_image[offset * 4 + 2] + x + 0.5f;
		int target2_y = vel_image[offset * 4 + 3] + y + 0.5f;

		float real_weight1 = 0;
		float real_weight2 = 0;
		if (blend_mode == 1)
		{
			real_weight1 = 1.0f;
			real_weight2 = 0.0f;
		}
		else if (blend_mode == 2)
		{
			real_weight1 = 0.0f;
			real_weight2 = 1.0f;
		}
		else
		{
			real_weight1 = weight1;
			real_weight2 = 1.0f - weight1;
		}

		int coord_x1 = clamp(target1_x, 0, width - 1);
		int coord_x2 = clamp(target2_x, 0, width - 1);
		int coord_y1 = clamp(target1_y, 0, height - 1);
		int coord_y2 = clamp(target2_y, 0, height - 1);

		for (int c = 0; c < nChannels; c++)
		{
			out[offset*nChannels + c] = real_weight1*image1[(coord_y1*width + coord_x1)*nChannels + c] + real_weight2*image2[(coord_y2*width + coord_x2)*nChannels + c];
		}


	}

	
	/***********  another implementation of neighbors begin ************/
	__global__
	void Compute_Neightbors_compute_neighbor_num_kernel(const int num, const float* coord_x, const float* coord_y, const float radius, 
					const int bucket_width, const int bucket_height,
					const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
					int* neighbor_num)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num)
			return;

		int bucket_idx = coord_in_which_bucket[idx];
		
		int w_idx = bucket_idx%bucket_width;
		int h_idx = bucket_idx/bucket_width;

		float cur_x = coord_x[idx];
		float cur_y = coord_y[idx];
		int cur_neigh_num = 0;
		
		float radius2 = radius*radius;
		
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{ 
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				int cur_bucket_idx = cur_h_idx*bucket_width+cur_w_idx;
				int cur_offset = bucket_stored_offset[cur_bucket_idx];
				for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
				{
					int cur_nei_idx = bucket_stored_index[cur_offset+iii];
					if(cur_nei_idx == idx)
						continue;
					float cur_nei_x = coord_x[cur_nei_idx];
					float cur_nei_y = coord_y[cur_nei_idx];
					
					float cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y);
					if(cur_dis2 <= radius2)
					{		
						cur_neigh_num ++;
					}
				}
			}
		}
		neighbor_num[idx] = cur_neigh_num;
	}
	
	__global__
	void Compute_Neightbors_fillin_neighbors_kernel(const int num, const float* coord_x, const float* coord_y, const float radius, 
					const int bucket_width, const int bucket_height, 
					const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
					const int* neighbor_offset, int* neighbor_index, float* neighbor_weight)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num)
			return;

		int bucket_idx = coord_in_which_bucket[idx];
		
		int w_idx = bucket_idx%bucket_width;
		int h_idx = bucket_idx/bucket_width;

		float cur_x = coord_x[idx];
		float cur_y = coord_y[idx];
		int cur_neigh_num = 0;
		int cur_neigh_offset = neighbor_offset[idx];
		
		float radius2 = radius*radius;
		
		for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
		{ 
			for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
			{
				int cur_bucket_idx = cur_h_idx*bucket_width+cur_w_idx;
				int cur_offset = bucket_stored_offset[cur_bucket_idx];
				for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
				{
					int cur_nei_idx = bucket_stored_index[cur_offset+iii];
					if(cur_nei_idx == idx)
						continue;
					float cur_nei_x = coord_x[cur_nei_idx];
					float cur_nei_y = coord_y[cur_nei_idx];

					float cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y);
					if(cur_dis2 <= radius2)
					{		
						neighbor_index[cur_neigh_offset+cur_neigh_num] = cur_nei_idx;
						neighbor_weight[cur_neigh_offset+cur_neigh_num] = _kernel_square(cur_dis2,radius2);
						cur_neigh_num ++;
					}
				}
			}
		}
		/*float sum_weight = 0;
		for(int i = 0;i < cur_neigh_num;i++)
			sum_weight += neighbor_weight[cur_neigh_offset+i];
		if(sum_weight != 0)
		{
			for(int i = 0;i < cur_neigh_num;i++)
				neighbor_weight[cur_neigh_offset+i] /= sum_weight;
		}	*/
	}
	
	__global__
	void solve_coeffs_part_various_neighbor_num_kernel(const int x_off, const int y_off, const int num, const int nChannels, const float* values, const int bucket_width, const int bucket_height, 						
									const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
									const int* neighbor_num, const int* neighbor_offset, const int* neighbor_index, const float* neighbor_weight, 
									float* coeffs)
	{
		int x = blockIdx.x*2 + x_off;
		int y = blockIdx.y*2 + y_off;
		int idx = threadIdx.x;
		if(x >= bucket_width || y >= bucket_height || idx >= nChannels)
			return;
			
		int bucket_idx = y*bucket_width+x;
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

	/*****************  another implementation of neighbors end ****************/
	
	void cu_Compute_Boundingbox(const int num, const float* coord_x, const float* coord_y, float boxmin[2], float boxmax[2])
	{
	
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,coord_x,boxmax[0]);
		ZQ_CUDA_BaseUtils::cu_Find_MAX_Value(num,coord_y,boxmax[1]);
		ZQ_CUDA_BaseUtils::cu_Find_MIN_Value(num,coord_x,boxmin[0]);
		ZQ_CUDA_BaseUtils::cu_Find_MIN_Value(num,coord_y,boxmin[1]);
	}

	void cu_Compute_coord_in_which_bucket(const int num, const float* coord_x, const float* coord_y, const float x_min, const float y_min,
			const float radius, const int bucket_width, const int bucket_height, int* coord_in_which_bucket)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		compute_coord_in_which_bucket_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,x_min,y_min,radius,bucket_width,bucket_height,coord_in_which_bucket);
	}
	
	

	void cu_Compute_Neightbors(const int num, const float* coord_x, const float* coord_y, const float radius, const int bucket_width, const int bucket_height, 
									  const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
									  int* neighbor_num, int* neighbor_index, float* neighbor_weight)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		Compute_Neightbors_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,radius,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,
			neighbor_num,neighbor_index,neighbor_weight);
	}
	
	/****************    another implementation of neighbors begin **********/
	
	void cu_Compute_Neightbors_various_neighbor_num(const int num, const float* coord_x, const float* coord_y, const float radius, const int bucket_width, const int bucket_height, 
									  const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
									  int* neighbor_num, int* neighbor_offset, int*& neighbor_index, float*& neighbor_weight)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		
		Compute_Neightbors_compute_neighbor_num_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,radius,bucket_width,bucket_height,
				bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,neighbor_num);
		
		ZQ_CUDA_BaseUtils::cu_Compute_bucket_stored_offset(num, neighbor_num, neighbor_offset);
		
		int total_neighbor_num;
		ZQ_CUDA_BaseUtils::cu_SUM(num, neighbor_num, total_neighbor_num);
		
		checkCudaErrors( cudaMalloc((void**)&neighbor_index,sizeof(int)*total_neighbor_num) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_weight,sizeof(float)*total_neighbor_num) );
		checkCudaErrors( cudaMemset(neighbor_index,0,sizeof(int)*total_neighbor_num) );
		checkCudaErrors( cudaMemset(neighbor_weight,0,sizeof(float)*total_neighbor_num) );
		
		Compute_Neightbors_fillin_neighbors_kernel<<<gridSize,blockSize>>>(num,coord_x,coord_y,radius,bucket_width,bucket_height,
				bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,neighbor_offset,neighbor_index,neighbor_weight);
	}
	
	/****************    another implementation of neighbors end **********/

	void cu_ScatteredInterpolation(const int num, const float* coord_x, const float* coord_y, const int nChannels, const float* values, const float radius, const int iterations,
								   const int out_width, const int out_height, float* out_images)
	{
		dim3 bS(1,1,1);
		dim3 gS(1,1,1);
	
		float boxmin[2],boxmax[2];
		cu_Compute_Boundingbox(num,coord_x,coord_y,boxmin,boxmax);
		
		//printf("(%.3f,%.3f),(%.3f,%.3f)\n",boxmin[0],boxmin[1],boxmax[0],boxmax[1]);

		/*float* h_values = new float[num*3];
		checkCudaErrors( cudaMemcpy(h_values,values,sizeof(float)*num*3,cudaMemcpyDeviceToHost) );
		for(int i = 0;i < num;i++)
		{
			printf("%6.2f %6.2f %6.2f\n",h_values[i*3+0],h_values[i*3+1],h_values[i*3+2]);
		}
		delete []h_values;
		int tmp;
		scanf("%d",&tmp);*/
		
		int bucket_width = (boxmax[0] - boxmin[0])/radius + 1;
		int bucket_height = (boxmax[1] - boxmin[1])/radius + 1;
		int bucket_num = bucket_width*bucket_height;
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
		
		cu_Compute_coord_in_which_bucket(num,coord_x,coord_y,boxmin[0],boxmin[1],radius,bucket_width,bucket_height,coord_in_which_bucket);
		
		ZQ_CUDA_BaseUtils::cu_Distribute_Bucket(num,bucket_num,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket);
		
		
		/*int* stored_num = new int[bucket_num];
		int* stored_offset = new int[bucket_num];
		checkCudaErrors( cudaMemcpy(stored_num,bucket_stored_num,sizeof(int)*bucket_num,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(stored_offset,bucket_stored_offset,sizeof(int)*bucket_num,cudaMemcpyDeviceToHost) );
		for(int i = 0;i < bucket_num;i++)
		{
			printf("bucket[%d]:%d,%d\n",i,stored_num[i],stored_offset[i]);
		}
		delete []stored_num;
		delete []stored_offset;*/
		
		int* neighbor_num = 0;
		int* neighbor_index = 0;
		float* neighbor_weight = 0;
		checkCudaErrors( cudaMalloc((void**)&neighbor_num,sizeof(int)*num) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_index,sizeof(int)*num*MAX_NEIGHBOR_NUM) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_weight,sizeof(float)*num*MAX_NEIGHBOR_NUM) );
		checkCudaErrors( cudaMemset(neighbor_num,0,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(neighbor_index,0,sizeof(int)*num*MAX_NEIGHBOR_NUM) );
		checkCudaErrors( cudaMemset(neighbor_weight,0,sizeof(float)*num*MAX_NEIGHBOR_NUM) );
		cu_Compute_Neightbors(num,coord_x,coord_y,radius,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,
			neighbor_num,neighbor_index,neighbor_weight);

		/*int* n_num = new int[num];
		checkCudaErrors( cudaMemcpy(n_num,neighbor_num,sizeof(int)*num,cudaMemcpyDeviceToHost) );
		int max_n_num = 0;
		for(int i = 0;i < num;i++)
		{
			if(max_n_num < n_num[i])
				max_n_num = n_num[i];
		}
		delete []n_num;
		printf("max_n_num=%d\n",max_n_num);*/
		
		

		float* coeffs = 0;
		checkCudaErrors( cudaMalloc((void**)&coeffs,sizeof(float)*num*nChannels) );
		checkCudaErrors( cudaMemset(coeffs,0,sizeof(float)*num*nChannels) );
		
		
		/*for(int it = 0;it < iterations;it++)
		{
			solve_coeffs_one_kernel<<<gS,bS>>>(num,nChannels,values,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
				neighbor_num,neighbor_index,neighbor_weight,coeffs);
		}*/
		
		/*float* h_coeffs = new float[num*3];
		checkCudaErrors( cudaMemcpy(h_coeffs,coeffs,sizeof(float)*num*3,cudaMemcpyDeviceToHost) );
		for(int i = 0;i < num;i++)
		{
			printf("%6.2f %6.2f %6.2f\n",h_coeffs[i*3+0],h_coeffs[i*3+1],h_coeffs[i*3+2]);
		}
		delete []h_coeffs;*/
		
		
		int x_off[4] = {0,0,1,1};
		int y_off[4] = {0,1,0,1};
		
		dim3 blockSize1(nChannels,1);
		dim3 gridSize1((bucket_width+1)/2,(bucket_height+1)/2);
		
		for(int it = 0;it < iterations;it++)
		{
			for(int iii = 0;iii < 4;iii++)
			{
				solve_coeffs_part_kernel<<<gridSize1,blockSize1>>>(x_off[iii],y_off[iii],num,nChannels,values,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
					neighbor_num,neighbor_index,neighbor_weight,coeffs);
			}
		}
		

		checkCudaErrors( cudaFree(neighbor_num) );
		checkCudaErrors( cudaFree(neighbor_index) );
		checkCudaErrors( cudaFree(neighbor_weight) );
		
		//splat_data_one_kernel<<<gS,bS>>>(num,coord_x,coord_y,radius,nChannels,coeffs,out_width,out_height,out_images);

		dim3 blockSize2(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize2((out_width+blockSize2.x-1)/blockSize2.x,(out_height+blockSize2.y-1)/blockSize2.y);
		compute_data_kernel<<<gridSize2,blockSize2>>>(out_width,out_height,boxmin[0],boxmin[1],radius,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,
					bucket_stored_index,coord_x,coord_y,coeffs,nChannels,out_images);
		
		
		
		checkCudaErrors( cudaFree(coeffs) );
		checkCudaErrors( cudaFree(bucket_stored_num) );
		checkCudaErrors( cudaFree(bucket_stored_offset) );
		checkCudaErrors( cudaFree(bucket_stored_index) );
		checkCudaErrors( cudaFree(coord_in_which_bucket) );
	}
	
	void cu_ScatteredInterpolation_various_neighbor_num(const int num, const float* coord_x, const float* coord_y, const int nChannels, const float* values, const float radius, const int iterations,
								   const int out_width, const int out_height, float* out_images)
	{
		dim3 bS(1,1,1);
		dim3 gS(1,1,1);
	
		float boxmin[2],boxmax[2];
		cu_Compute_Boundingbox(num,coord_x,coord_y,boxmin,boxmax);
		
		
		int bucket_width = (boxmax[0] - boxmin[0])/radius + 1;
		int bucket_height = (boxmax[1] - boxmin[1])/radius + 1;
		int bucket_num = bucket_width*bucket_height;
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
		
		cu_Compute_coord_in_which_bucket(num,coord_x,coord_y,boxmin[0],boxmin[1],radius,bucket_width,bucket_height,coord_in_which_bucket);
		
		ZQ_CUDA_BaseUtils::cu_Distribute_Bucket(num,bucket_num,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket);
		
		
		int* neighbor_num = 0;
		int* neighbor_offset = 0;
		int* neighbor_index = 0;
		float* neighbor_weight = 0;
		checkCudaErrors( cudaMalloc((void**)&neighbor_num,sizeof(int)*num) );
		checkCudaErrors( cudaMalloc((void**)&neighbor_offset,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(neighbor_num,0,sizeof(int)*num) );
		checkCudaErrors( cudaMemset(neighbor_offset,0,sizeof(int)*num) );
	
		cu_Compute_Neightbors_various_neighbor_num(num,coord_x,coord_y,radius,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,
			neighbor_num,neighbor_offset,neighbor_index,neighbor_weight);
		

		float* coeffs = 0;
		checkCudaErrors( cudaMalloc((void**)&coeffs,sizeof(float)*num*nChannels) );
		checkCudaErrors( cudaMemset(coeffs,0,sizeof(float)*num*nChannels) );
		
		
		int x_off[4] = {0,0,1,1};
		int y_off[4] = {0,1,0,1};
		
		dim3 blockSize1(4,1);
		dim3 gridSize1((bucket_width+1)/2,(bucket_height+1)/2);
		
		for(int it = 0;it < iterations;it++)
		{
			for(int iii = 0;iii < 4;iii++)
			{
				solve_coeffs_part_various_neighbor_num_kernel<<<gridSize1,blockSize1>>>(x_off[iii],y_off[iii],num,nChannels,values,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
					neighbor_num,neighbor_offset,neighbor_index,neighbor_weight,coeffs);
			}
		}
		

		checkCudaErrors( cudaFree(neighbor_num) );
		checkCudaErrors( cudaFree(neighbor_offset) );
		checkCudaErrors( cudaFree(neighbor_index) );
		checkCudaErrors( cudaFree(neighbor_weight) );
		
		
		dim3 blockSize2(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize2((out_width+blockSize2.x-1)/blockSize2.x,(out_height+blockSize2.y-1)/blockSize2.y);
		compute_data_kernel<<<gridSize2,blockSize2>>>(out_width,out_height,boxmin[0],boxmin[1],radius,bucket_width,bucket_height,bucket_stored_num,bucket_stored_offset,
					bucket_stored_index,coord_x,coord_y,coeffs,nChannels,out_images);
		
		
		
		
		checkCudaErrors( cudaFree(coeffs) );
		checkCudaErrors( cudaFree(bucket_stored_num) );
		checkCudaErrors( cudaFree(bucket_stored_offset) );
		checkCudaErrors( cudaFree(bucket_stored_index) );
		checkCudaErrors( cudaFree(coord_in_which_bucket) );
	}


	void cu_BlendTwoImages(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float weight1, 
				const int skip, const float radius, const int iterations, float* out_image, const bool various_neighbor_num, const int sample_mode, const int blend_mode)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		float* coord_x = 0;
		float* coord_y = 0;
		float* values = 0;
		int seed_width = width/skip;
		int seed_height = height/skip;
		int num = seed_width*seed_height;
		checkCudaErrors( cudaMalloc((void**)&coord_x,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_y,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&values,sizeof(float)*num*4) );
		checkCudaErrors( cudaMemset(coord_x,0,sizeof(float)*num) );
		checkCudaErrors( cudaMemset(coord_y,0,sizeof(float)*num) );
		checkCudaErrors( cudaMemset(values,0,sizeof(float)*num*4) );
		
		Compute_coords_and_values_kernel<<<gridSize,blockSize>>>(seed_width,seed_height,width,height,skip,u,v,weight1,coord_x,coord_y,values);
		
		float* vel_image = 0;
		checkCudaErrors( cudaMalloc((void**)&vel_image,sizeof(float)*width*height*4) );
		checkCudaErrors( cudaMemset(vel_image,0,sizeof(float)*width*height*4) );
	
		if(various_neighbor_num)
			cu_ScatteredInterpolation_various_neighbor_num(num, coord_x, coord_y, 4, values, radius, iterations, width, height, vel_image);
		else
			cu_ScatteredInterpolation(num, coord_x, coord_y, 4, values, radius, iterations, width, height, vel_image);
		
		
		if(sample_mode == 0)
			warp_and_blend_kernel<<<gridSize,blockSize>>>(width, height, nChannels, image1, image2, vel_image, weight1, out_image, blend_mode);
		else if (sample_mode == 1)
			warp_and_blend_bicubic_kernel<<<gridSize,blockSize>>>(width, height, nChannels, image1, image2, vel_image, weight1, out_image, blend_mode);
		else
			warp_and_blend_interger_kernel<< <gridSize, blockSize >> >(width, height, nChannels, image1, image2, vel_image, weight1, out_image, blend_mode);
		
		checkCudaErrors( cudaFree(coord_x) );
		checkCudaErrors( cudaFree(coord_y) );
		checkCudaErrors( cudaFree(vel_image) );
	}

	void cu_InterpolateVelocityByMedFilt_4channels(const int width, const int height, const float* u, const float* v, const float weight1, float* vel_image)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

		float* tmp_vel_image = 0;
		checkCudaErrors(cudaMalloc((void**)&tmp_vel_image, sizeof(float)*width*height * 4));
		checkCudaErrors(cudaMemset(tmp_vel_image, 0, sizeof(float)*width*height * 4));
		bool* keep_mask = 0;
		checkCudaErrors(cudaMalloc((void**)&keep_mask, sizeof(bool)*width*height));
		checkCudaErrors(cudaMemset(keep_mask, 0, sizeof(bool)*width*height));

		forward_move_velocity_kernel << <gridSize, blockSize >> >(width, height, weight1, u, v, tmp_vel_image, keep_mask);
		ZQ_CUDA_ImageProcessing2D::MedianFilterWithMask5x5_Kernel << <gridSize, blockSize >> >(vel_image, tmp_vel_image, width, height, 4, keep_mask);
		
		checkCudaErrors(cudaFree(tmp_vel_image));
		checkCudaErrors(cudaFree(keep_mask));
		tmp_vel_image = 0;
		keep_mask = 0;
	}

	void cu_BlendTwoImagesByMedFilt(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float weight1,
		 float* out_image,  const int sample_mode, const int blend_mode)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

		float* vel_image = 0;
		checkCudaErrors(cudaMalloc((void**)&vel_image, sizeof(float)*width*height * 4));
		checkCudaErrors(cudaMemset(vel_image, 0, sizeof(float)*width*height * 4));

		cu_InterpolateVelocityByMedFilt_4channels(width, height, u, v, weight1, vel_image);

		if (sample_mode == 0)
			warp_and_blend_kernel << <gridSize, blockSize >> >(width, height, nChannels, image1, image2, vel_image, weight1, out_image, blend_mode);
		else if (sample_mode == 1)
			warp_and_blend_bicubic_kernel << <gridSize, blockSize >> >(width, height, nChannels, image1, image2, vel_image, weight1, out_image, blend_mode);
		else
			warp_and_blend_interger_kernel << <gridSize, blockSize >> >(width, height, nChannels, image1, image2, vel_image, weight1, out_image, blend_mode);

		checkCudaErrors(cudaFree(vel_image));
	}

	extern "C"
	float Cutil_ScatteredInterpolation(const int num, const float* coord_x, const float* coord_y, const int nChannels, const float* values, const float radius, const int iterations,
								   const int out_width, const int out_height, float* out_images, bool various_neighbor_num)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* coord_x_d = 0;
		float* coord_y_d = 0;
		float* values_d = 0; 
		checkCudaErrors( cudaMalloc((void**)&coord_x_d,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&coord_y_d,sizeof(float)*num) );
		checkCudaErrors( cudaMalloc((void**)&values_d,sizeof(float)*num*nChannels) );
		checkCudaErrors( cudaMemcpy(coord_x_d,coord_x,sizeof(float)*num,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(coord_y_d,coord_y,sizeof(float)*num,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(values_d,values,sizeof(float)*num*nChannels,cudaMemcpyHostToDevice) );

		float* out_images_d = 0;
		checkCudaErrors( cudaMalloc((void**)&out_images_d,sizeof(float)*out_width*out_height*nChannels) );
		checkCudaErrors( cudaMemset(out_images_d,0,sizeof(float)*out_width*out_height*nChannels) );

		if(various_neighbor_num)
			cu_ScatteredInterpolation_various_neighbor_num(num,coord_x_d,coord_y_d,nChannels,values_d,radius,iterations,out_width,out_height,out_images_d);
		else
			cu_ScatteredInterpolation(num,coord_x_d,coord_y_d,nChannels,values_d,radius,iterations,out_width,out_height,out_images_d);
		
		checkCudaErrors( cudaMemcpy(out_images,out_images_d,sizeof(float)*out_width*out_height*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(coord_x_d) );
		checkCudaErrors( cudaFree(coord_y_d) );
		checkCudaErrors( cudaFree(values_d) );
		checkCudaErrors( cudaFree(out_images_d) );

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float Cutil_BlendTwoImages(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float weight1, 
			const int skip, const float radius, const int iterations, float* out_image, const bool various_neighbor_num, const int sample_mode, const int blend_mode)
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
		float* out_image_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&image1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&image2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&out_image_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(image1_d,image1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(image2_d,image2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(out_image_d,0,sizeof(float)*width*height*nChannels) );
		
		cu_BlendTwoImages(width, height, nChannels, image1_d, image2_d, u_d, v_d, weight1, skip, radius, iterations, out_image_d, various_neighbor_num,sample_mode,blend_mode);
		
		checkCudaErrors( cudaMemcpy(out_image,out_image_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(image1_d) );
		checkCudaErrors( cudaFree(image2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(out_image_d) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float Cutil_BlendTwoImagesByMedFilt(const int width, const int height, const int nChannels, const float* image1, const float* image2, const float* u, const float* v, const float weight1,
		float* out_image, const int sample_mode, const int blend_mode)
	{
		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		float* image1_d = 0;
		float* image2_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* out_image_d = 0;

		checkCudaErrors(cudaMalloc((void**)&image1_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&image2_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&u_d, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&v_d, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&out_image_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemcpy(image1_d, image1, sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(image2_d, image2, sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(u_d, u, sizeof(float)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(v_d, v, sizeof(float)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(out_image_d, 0, sizeof(float)*width*height*nChannels));

		cu_BlendTwoImagesByMedFilt(width, height, nChannels, image1_d, image2_d, u_d, v_d, weight1, out_image_d, sample_mode, blend_mode);

		checkCudaErrors(cudaMemcpy(out_image, out_image_d, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(image1_d));
		checkCudaErrors(cudaFree(image2_d));
		checkCudaErrors(cudaFree(u_d));
		checkCudaErrors(cudaFree(v_d));
		checkCudaErrors(cudaFree(out_image_d));

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
}

#endif