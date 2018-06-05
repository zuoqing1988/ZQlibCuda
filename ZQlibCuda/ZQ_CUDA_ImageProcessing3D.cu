#ifndef _ZQ_CUDA_IMAGE_PROCESSING_3D_CU_
#define _ZQ_CUDA_IMAGE_PROCESSING_3D_CU_

#include "ZQ_CUDA_ImageProcessing3D.cuh"

namespace ZQ_CUDA_ImageProcessing3D
{	
	texture<float,3,cudaReadModeElementType> tex_img_1channel;
	texture<float4,3,cudaReadModeElementType> tex_img_4channel;
	
	/***************************************************************************/
	__global__
	void WarpImage_1channel_Kernel(float* warpIm2, const float* Im1, const float* u, const float* v, const float* w, const int width, const int height, const int depth)
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

			float target_x = x + u[offset];
			float target_y = y + v[offset];
			float target_z = z + w[offset];

			if(target_x < 0 || target_x > width-1 || target_y < 0 || target_y > height-1 || target_z < 0 || target_z > depth-1)
			{
				warpIm2[offset] = Im1[offset];
			}
			else
			{
				float coord_x = (target_x+0.5f)/width;
				float coord_y = (target_y+0.5f)/height;
				float coord_z = (target_z+0.5f)/depth;

				warpIm2[offset] = tex3D(tex_img_1channel,coord_x,coord_y,coord_z);
			}
		}
	}
	
	__global__
	void WarpImage_4channel_Kernel(float4* warpIm2, const float4* Im1, const float* u, const float* v, const float* w, const int width, const int height, const int depth)
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

			float target_x = x + u[offset];
			float target_y = y + v[offset];
			float target_z = z + w[offset];

			if(target_x < 0 || target_x > width-1 || target_y < 0 || target_y > height-1 || target_z < 0 || target_z > depth-1)
			{
				warpIm2[offset] = Im1[offset];
			}
			else
			{
				float coord_x = (target_x+0.5f)/width;
				float coord_y = (target_y+0.5f)/height;
				float coord_z = (target_z+0.5f)/depth;

				warpIm2[offset] = tex3D(tex_img_4channel,coord_x,coord_y,coord_z);
			}
		}
	}
	
	__global__
	void WarpImage_Trilinear_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w, 
										const int width, const int height, const int depth, const int nChannels)
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

			float target_x = x + u[offset];
			float target_y = y + v[offset];
			float target_z = z + w[offset];

			if(target_x < 0 || target_x > width-1 || target_y < 0 || target_y > height-1 || target_z < 0 || target_z > depth-1)
			{
				for(int c = 0;c < nChannels;c++)
					warpIm2[offset*nChannels+c] = Im1[offset*nChannels+c];
			}
			else
			{
				float coord_x = (target_x+0.5f)/width;
				float coord_y = (target_y+0.5f)/height;
				float coord_z = (target_z+0.5f)/depth;

				for(int c = 0;c < nChannels;c++)
				{
					int x0 = floor(coord_x);
					int x1 = x0+1;
					int y0 = floor(coord_y);
					int y1 = y0+1;
					int z0 = floor(coord_z);
					int z1 = z0+1;

					float sx = coord_x-x0;
					float sy = coord_y-y0;
					float sz = coord_z-z0;

					float val = 0.0f;
					int real_x0 = clamp(x0,0,width-1);
					int real_x1 = clamp(x1,0,width-1);
					int real_y0 = clamp(y0,0,height-1);
					int real_y1 = clamp(y1,0,height-1);
					int real_z0 = clamp(z0,0,depth-1);
					int real_z1 = clamp(z1,0,depth-1);
					
					val += Im2[(real_z0*height*width+real_y0*width+real_x0)*nChannels+c]*(1.0f-sx)*(1.0f-sy)*(1.0f-sz);
					val += Im2[(real_z0*height*width+real_y1*width+real_x0)*nChannels+c]*(1.0f-sx)*      sy *(1.0f-sz);
					val += Im2[(real_z0*height*width+real_y0*width+real_x1)*nChannels+c]*      sx *(1.0f-sy)*(1.0f-sz);
					val += Im2[(real_z0*height*width+real_y1*width+real_x1)*nChannels+c]*      sx *      sy *(1.0f-sz);
					val += Im2[(real_z1*height*width+real_y0*width+real_x0)*nChannels+c]*(1.0f-sx)*(1.0f-sy)*      sz ;
					val += Im2[(real_z1*height*width+real_y1*width+real_x0)*nChannels+c]*(1.0f-sx)*      sy *      sz ;
					val += Im2[(real_z1*height*width+real_y0*width+real_x1)*nChannels+c]*      sx *(1.0f-sy)*      sz ;
					val += Im2[(real_z1*height*width+real_y1*width+real_x1)*nChannels+c]*      sx *      sy *      sz ;
					warpIm2[offset*nChannels+c] = val;
				}
			}
		}
	}
	
	
	__global__
	void WarpImage_Tricubic_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w, 
							const int width, const int height, const int depth, const int nChannels)
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

			float target_x = x + u[offset];
			float target_y = y + v[offset];
			float target_z = z + w[offset];

			if(target_x < 0 || target_x > width-1 || target_y < 0 || target_y > height-1 || target_z < 0 || target_z > depth-1)
			{
				for(int c = 0;c < nChannels;c++)
					warpIm2[offset*nChannels+c] = Im1[offset*nChannels+c];
			}
			else
			{
				float coord_x = target_x;
				float coord_y = target_y;
				float coord_z = target_z;

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
								data_x[i] = Im2[(tmp_z*height*width+tmp_y*width+tmp_x)*nChannels+c];
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

					warpIm2[offset*nChannels+c] = a0 + fz*(a1 + fz*(a2+fz*a3));
				}
			}
		}
	}
	
	__global__
	void ResizeImage_1channel_Kernel(float* dst, const int dst_width, const int dst_height, const int dst_depth)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= dst_width || y >= dst_height)
			return ;

		for(int z = 0;z < dst_depth;z++)
		{
			int offset = z*dst_height*dst_width+y*dst_width+x;

			float coord_x = (x+0.5f)/dst_width;
			float coord_y = (y+0.5f)/dst_height;
			float coord_z = (z+0.5f)/dst_depth;

			dst[offset] = tex3D(tex_img_1channel,coord_x,coord_y,coord_z);
		}
	}

	__global__
	void ResizeImage_4channel_Kernel(float4* dst, const int dst_width, const int dst_height, const int dst_depth)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= dst_width || y >= dst_height)
			return ;

		for(int z = 0;z < dst_depth;z++)
		{
			int offset = z*dst_height*dst_width+y*dst_width+x;

			float coord_x = (x+0.5f)/dst_width;
			float coord_y = (y+0.5f)/dst_height;
			float coord_z = (z+0.5f)/dst_depth;

			dst[offset] = tex3D(tex_img_4channel,coord_x,coord_y,coord_z);
		}
	}

	__global__
	void ResizeImage_Trilinear_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int src_depth,
					const int dst_width, const int dst_height, const int dst_depth, const int nChannels)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= dst_width || y >= dst_height)
			return ;

		for(int z = 0;z < dst_depth;z++)
		{
			int offset = z*dst_height*dst_width+y*dst_width+x;

			float coord_x = (x+0.5f)/dst_width*src_width - 0.5f;
			float coord_y = (y+0.5f)/dst_height*src_height - 0.5f;
			float coord_z = (z+0.5f)/dst_depth*src_depth - 0.5f;

			int x0 = floor(coord_x);
			int x1 = x0+1;
			int y0 = floor(coord_y);
			int y1 = y0+1;
			int z0 = floor(coord_z);
			int z1 = z0+1;

			float sx = coord_x-x0;
			float sy = coord_y-y0;
			float sz = coord_z-z0;

			int real_x0 = clamp(x0,0,src_width-1);
			int real_x1 = clamp(x1,0,src_width-1);
			int real_y0 = clamp(y0,0,src_height-1);
			int real_y1 = clamp(y1,0,src_height-1);
			int real_z0 = clamp(z0,0,src_depth-1);
			int real_z1 = clamp(z1,0,src_depth-1);

			for(int c = 0;c < nChannels;c++)
			{
				float val = 0;
				val += src[(real_z0*src_height*src_width+real_y0*src_width+real_x0)*nChannels+c]*(1.0f-sx)*(1.0f-sy)*(1.0f-sz);
				val += src[(real_z0*src_height*src_width+real_y0*src_width+real_x1)*nChannels+c]*      sx *(1.0f-sy)*(1.0f-sz);
				val += src[(real_z0*src_height*src_width+real_y1*src_width+real_x0)*nChannels+c]*(1.0f-sx)*      sy *(1.0f-sz);
				val += src[(real_z0*src_height*src_width+real_y1*src_width+real_x1)*nChannels+c]*      sx *      sy *(1.0f-sz);
				val += src[(real_z1*src_height*src_width+real_y0*src_width+real_x0)*nChannels+c]*(1.0f-sx)*(1.0f-sy)*      sz ;
				val += src[(real_z1*src_height*src_width+real_y0*src_width+real_x1)*nChannels+c]*      sx *(1.0f-sy)*      sz ;
				val += src[(real_z1*src_height*src_width+real_y1*src_width+real_x0)*nChannels+c]*(1.0f-sx)*      sy *      sz ;
				val += src[(real_z1*src_height*src_width+real_y1*src_width+real_x1)*nChannels+c]*      sx *      sy *      sz ;

				dst[offset*nChannels+c] = val;
			}
		}
	}
	
	__global__
	void ResizeImage_Tricubic_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int src_depth, 
								const int dst_width, const int dst_height, const int dst_depth, const int nChannels)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x+tx;
		int y = by*blockDim.y+ty;
		if(x >= dst_width || y >= dst_height)
			return ;

		for(int z = 0;z < dst_depth;z++)
		{
			int offset = z*dst_height*dst_width+y*dst_width+x;

			float coord_x = (x+0.5)/dst_width*src_width - 0.5;
			float coord_y = (y+0.5)/dst_height*src_height - 0.5;
			float coord_z = (z+0.5)/dst_depth*src_depth - 0.5;

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
					int tmp_z = clamp(iz-1+k,0,src_depth-1);
					for(int j = 0;j < 4;j++)
					{
						float data_x[4];
						int tmp_y = clamp(iy-1+j,0,src_height-1);
						for(int i = 0;i < 4;i++)
						{
							int tmp_x = clamp(ix-1+i,0,src_width-1);
							data_x[i] = src[(tmp_z*src_height*src_width+tmp_y*src_width+tmp_x)*nChannels+c];
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

				dst[offset*nChannels+c] = a0 + fz*(a1 + fz*(a2+fz*a3));
			}
		}
	}

	__global__
	void Addwith_Kernel(float* in_out_put, const float* other, const float weight, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			for(int c = 0;c < nChannels;c++)
			{
				in_out_put[offset*nChannels+c] += other[offset*nChannels+c]*weight;
			}
		}
	}
	
	__global__
	void Add_Im1_weight1_Im2_weight2_Kernel(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2, 
											const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			for(int c = 0;c < nChannels;c++)
			{
				output[offset*nChannels+c] = Im1[offset*nChannels+c]*weight1 + Im2[offset*nChannels+c]*weight2;
			}
		}
	}
	
	__global__
	void Imfilter_h_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float pfilter1D[5] = {0.02,0.11,0.74,0.11,0.02};
			int fsize = 2;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int xx = clamp(x+l,0,width-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(z*height*width+y*width+xx)*nChannels+c]*pfilter1D[l+fsize];
			}
		}
	}
	
	__global__
	void Imfilter_v_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float pfilter1D[5] = {0.02,0.11,0.74,0.11,0.02};
			int fsize = 2;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int yy = clamp(y+l,0,height-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(z*height*width+yy*width+x)*nChannels+c]*pfilter1D[l+fsize];
			}
		}
	}
	
	__global__
	void Imfilter_d_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float pfilter1D[5] = {0.02,0.11,0.74,0.11,0.02};
			int fsize = 2;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int zz = clamp(z+l,0,depth-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(zz*height*width+y*width+x)*nChannels+c]*pfilter1D[l+fsize];
			}
		}
	}
	
	__global__
	void Imfilter_h_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int xx = clamp(x+l,0,width-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(z*height*width+y*width+xx)*nChannels+c]*filter[l+fsize];
			}
		}
	}
	
	__global__
	void Imfilter_v_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int yy = clamp(y+l,0,height-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(z*height*width+yy*width+x)*nChannels+c]*filter[l+fsize];
			}
		}
	}
	
	__global__
	void Imfilter_d_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int zz = clamp(z+l,0,depth-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(zz*height*width+y*width+x)*nChannels+c]*filter[l+fsize];
			}
		}
	}
	
	__global__
	void Derivative_x_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
			
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float pfilter1D[5] = {1.0/12,-8.0/12,0,8.0/12,-1.0/12};
			int fsize = 2;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int xx = clamp(x+l,0,width-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(z*height*width+y*width+xx)*nChannels+c]*pfilter1D[l+fsize];
			}
		}
	}
	
	__global__
	void Derivative_y_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
			
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float pfilter1D[5] = {1.0/12,-8.0/12,0,8.0/12,-1.0/12};
			int fsize = 2;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int yy = clamp(y+l,0,height-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(z*height*width+yy*width+x)*nChannels+c]*pfilter1D[l+fsize];
			}
		}
	}
	
	__global__
	void Derivative_z_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
			
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float pfilter1D[5] = {1.0/12,-8.0/12,0,8.0/12,-1.0/12};
			int fsize = 2;

			for(int c = 0;c < nChannels;c++)
				output[offset*nChannels+c] = 0;

			for(int l = -fsize; l <= fsize;l++)
			{
				int zz = clamp(z+l,0,depth-1);
				for(int c = 0;c < nChannels;c++)
					output[offset*nChannels+c] += input[(zz*height*width+y*width+x)*nChannels+c]*pfilter1D[l+fsize];
			}
		}
	}
	
	__global__
	void Laplacian_Kernel(float* output, const float* input, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			for(int c = 0;c < nChannels;c++)
			{
				float value = 0;
				if(x == 0)
				{
					value += input[(offset+1)*nChannels+c] - input[offset*nChannels+c];
				}
				else if(x == width-1)
				{
					value += input[(offset-1)*nChannels+c] - input[offset*nChannels+c];
				}
				else 
				{
					value += input[(offset+1)*nChannels+c] + input[(offset-1)*nChannels+c] - 2*input[offset*nChannels+c];
				}

				if(y == 0)
				{
					value += input[(offset+width)*nChannels+c] - input[offset*nChannels+c];
				}
				else if(y == height-1)
				{
					value += input[(offset-width)*nChannels+c] - input[offset*nChannels+c];
				}
				else
				{
					value += input[(offset+width)*nChannels+c] + input[(offset-width)*nChannels+c] - 2*input[offset*nChannels+c];
				}
				
				if(z == 0)
				{
					value += input[(offset+height*width)*nChannels+c] - input[offset*nChannels+c];
				}
				else if(z == depth-1)
				{
					value += input[(offset-height*width)*nChannels+c] - input[offset*nChannels+c];
				}
				else
				{
					value += input[(offset+height*width)*nChannels+c] + input[(offset-height*width)*nChannels+c] - 2*input[offset*nChannels+c];
				}

				output[offset*nChannels+c] = value;
			}
		}
	}
	
	/******************************************************/

	extern "C"
	float WarpImage3D_Trilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w,
							const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		if(nChannels == 1)
		{
			tex_img_1channel.normalized = true;                      
			tex_img_1channel.filterMode = cudaFilterModeLinear;     
			tex_img_1channel.addressMode[0] = cudaAddressModeClamp; 
			tex_img_1channel.addressMode[1] = cudaAddressModeClamp;
			tex_img_1channel.addressMode[2] = cudaAddressModeClamp;

			cudaArray* Im2_array = 0;
			float* warpIm2_d = 0;
			float* u_d = 0;
			float* v_d = 0;
			float* w_d = 0;
			float* Im1_d = 0;

			checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );

			checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*depth) );

			cudaChannelFormatDesc channelDescf = cudaCreateChannelDesc<float>();
			
			cudaExtent texSize = make_cudaExtent(width,height,depth);
			checkCudaErrors( cudaMalloc3DArray(&Im2_array, &channelDescf, texSize) );

			// copy data to 3D array
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr   = make_cudaPitchedPtr((void*)Im2, texSize.width*sizeof(float), texSize.width, texSize.height);
			copyParams.dstArray = Im2_array;
			copyParams.extent   = texSize;
			copyParams.kind     = cudaMemcpyHostToDevice;
			checkCudaErrors( cudaMemcpy3D(&copyParams) );

			checkCudaErrors( cudaBindTextureToArray(tex_img_1channel,Im2_array,channelDescf) );

			WarpImage_1channel_Kernel<<<gridSize,blockSize>>>(warpIm2_d,Im1_d,u_d,v_d,w_d,width,height,depth);

			checkCudaErrors( cudaUnbindTexture(tex_img_1channel) );


			checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaFree(warpIm2_d) );
			checkCudaErrors( cudaFree(Im1_d) );
			checkCudaErrors( cudaFree(u_d) );
			checkCudaErrors( cudaFree(v_d) );
			checkCudaErrors( cudaFree(w_d) );
			checkCudaErrors( cudaFreeArray(Im2_array) );

			warpIm2_d = 0;
			Im1_d = 0;
			u_d = 0;
			v_d = 0;
			w_d = 0;
			Im2_array = 0;
		}
		else if(nChannels == 4)
		{
			tex_img_4channel.normalized = true;                      
			tex_img_4channel.filterMode = cudaFilterModeLinear;     
			tex_img_4channel.addressMode[0] = cudaAddressModeClamp; 
			tex_img_4channel.addressMode[1] = cudaAddressModeClamp;
			tex_img_4channel.addressMode[2] = cudaAddressModeClamp;

			cudaArray* Im2_array = 0;
			float4* warpIm2_d = 0;
			float* u_d = 0;
			float* v_d = 0;
			float* w_d = 0;
			float4* Im1_d = 0;

			checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*4) );
			checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*4,cudaMemcpyHostToDevice) );

			checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*4) );
			checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*depth*4) );

			
			cudaChannelFormatDesc channelDescf4 = cudaCreateChannelDesc<float4>();
			
			cudaExtent texSize = make_cudaExtent(width,height,depth);
			checkCudaErrors( cudaMalloc3DArray(&Im2_array, &channelDescf4, texSize) );

			// copy data to 3D array
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr   = make_cudaPitchedPtr((void*)Im2, texSize.width*sizeof(float4), texSize.width, texSize.height);
			copyParams.dstArray = Im2_array;
			copyParams.extent   = texSize;
			copyParams.kind     = cudaMemcpyHostToDevice;
			checkCudaErrors( cudaMemcpy3D(&copyParams) );

			checkCudaErrors( cudaBindTextureToArray(tex_img_4channel,Im2_array,channelDescf4) );

			WarpImage_4channel_Kernel<<<gridSize,blockSize>>>(warpIm2_d,Im1_d,u_d,v_d,w_d,width,height,depth);

			checkCudaErrors( cudaUnbindTexture(tex_img_4channel) );

			checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*4,cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaFree(warpIm2_d) );
			checkCudaErrors( cudaFree(Im1_d) );
			checkCudaErrors( cudaFree(u_d) );
			checkCudaErrors( cudaFree(v_d) );
			checkCudaErrors( cudaFree(w_d) );
			checkCudaErrors( cudaFreeArray(Im2_array) );

			warpIm2_d = 0;
			Im1_d = 0;
			u_d = 0;
			v_d = 0;
			w_d = 0;
			Im2_array = 0;
		}
		else
		{
			float* warpIm2_d = 0;
			float* u_d = 0;
			float* v_d = 0;
			float* w_d = 0;
			float* Im1_d = 0;
			float* Im2_d = 0;

			checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
			checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
			checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );

			checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*depth*nChannels) );
			checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*4,cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*4,cudaMemcpyHostToDevice) );

			WarpImage_Trilinear_Kernel<<<gridSize,blockSize>>>(warpIm2_d,Im1_d,Im2_d,u_d,v_d,w_d,width,height,depth,nChannels);

			checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*4,cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaFree(warpIm2_d) );
			checkCudaErrors( cudaFree(u_d) );
			checkCudaErrors( cudaFree(v_d) );
			checkCudaErrors( cudaFree(w_d) );
			checkCudaErrors( cudaFree(Im1_d) );
			checkCudaErrors( cudaFree(Im2_d) );
			warpIm2_d = 0;
			u_d = 0;
			v_d = 0;
			w_d = 0;
			Im1_d = 0;
			Im2_d = 0;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float WarpImage3D_Tricubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const float* w,
							const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* warpIm2_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );

		checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*4,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*4,cudaMemcpyHostToDevice) );

		WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2_d,Im1_d,Im2_d,u_d,v_d,w_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*4,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		warpIm2_d = 0;
		u_d = 0;
		v_d = 0;
		w_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float ResizeImage3D_Trilinear(float* dst, const float* src, const int src_width, const int src_height, const int src_depth,
							const int dst_width, const int dst_height, const int dst_depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((dst_width-1)/blockSize.x+1,(dst_height-1)/blockSize.y+1);

		if(nChannels == 1)
		{
			tex_img_1channel.normalized = true;                      
			tex_img_1channel.filterMode = cudaFilterModeLinear;     
			tex_img_1channel.addressMode[0] = cudaAddressModeClamp; 
			tex_img_1channel.addressMode[1] = cudaAddressModeClamp;
			tex_img_1channel.addressMode[2] = cudaAddressModeClamp;

			cudaArray* src_array = 0;
			float* dst_d = 0;
		
			checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*dst_width*dst_height*dst_depth) );
			checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*dst_width*dst_height*dst_depth) );

			cudaExtent texSize = make_cudaExtent(src_width,src_height,src_depth);

			cudaChannelFormatDesc channelDescf = cudaCreateChannelDesc<float>();
			checkCudaErrors( cudaMalloc3DArray(&src_array, &channelDescf, texSize) );

			// copy data to 3D array
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr   = make_cudaPitchedPtr((void*)src, texSize.width*sizeof(float), texSize.width, texSize.height);
			copyParams.dstArray = src_array;
			copyParams.extent   = texSize;
			copyParams.kind     = cudaMemcpyHostToDevice;
			checkCudaErrors( cudaMemcpy3D(&copyParams) );

			checkCudaErrors( cudaBindTextureToArray(tex_img_1channel,src_array,channelDescf) );

			ResizeImage_1channel_Kernel<<<gridSize,blockSize>>>(dst_d,dst_width,dst_height,dst_depth);

			checkCudaErrors( cudaUnbindTexture(tex_img_1channel) );


			checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*dst_width*dst_height*dst_depth,cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaFree(dst_d) );
			checkCudaErrors( cudaFreeArray(src_array) );

			dst_d = 0;
			src_array = 0;
		}
		else if(nChannels == 4)
		{
			tex_img_4channel.normalized = true;                      
			tex_img_4channel.filterMode = cudaFilterModeLinear;     
			tex_img_4channel.addressMode[0] = cudaAddressModeClamp; 
			tex_img_4channel.addressMode[1] = cudaAddressModeClamp;
			tex_img_4channel.addressMode[2] = cudaAddressModeClamp;

			cudaArray* src_array = 0;
			float4* dst_d = 0;

			checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*dst_width*dst_height*dst_depth*4) );
			checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*dst_width*dst_height*dst_depth*4) );

			cudaExtent texSize = make_cudaExtent(src_width,src_height,src_depth);

			cudaChannelFormatDesc channelDescf4 = cudaCreateChannelDesc<float4>();
			checkCudaErrors( cudaMalloc3DArray(&src_array, &channelDescf4, texSize) );

			// copy data to 3D array
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr   = make_cudaPitchedPtr((void*)src, texSize.width*sizeof(float4), texSize.width, texSize.height);
			copyParams.dstArray = src_array;
			copyParams.extent   = texSize;
			copyParams.kind     = cudaMemcpyHostToDevice;
			checkCudaErrors( cudaMemcpy3D(&copyParams) );

			checkCudaErrors( cudaBindTextureToArray(tex_img_4channel,src_array,channelDescf4) );

			ResizeImage_4channel_Kernel<<<gridSize,blockSize>>>(dst_d,dst_width,dst_height,dst_depth);

			checkCudaErrors( cudaUnbindTexture(tex_img_4channel) );


			checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*dst_width*dst_height*dst_depth*4,cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaFree(dst_d) );
			checkCudaErrors( cudaFreeArray(src_array) );

			dst_d = 0;
			src_array = 0;
		}
		else
		{
			float* dst_d = 0;
			float* src_d = 0;

			checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*dst_width*dst_height*dst_depth*nChannels) );
			checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*src_width*src_height*src_depth*nChannels) );
			checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*dst_width*dst_height*dst_depth*nChannels) );
			checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*src_width*src_height*src_depth*nChannels,cudaMemcpyHostToDevice) );

			ResizeImage_Trilinear_Kernel<<<gridSize,blockSize>>>(dst_d,src_d,src_width,src_height,src_depth,dst_width,dst_height,dst_depth,nChannels);

			checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*dst_width*dst_height*dst_depth*nChannels,cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaFree(dst_d) );
			checkCudaErrors( cudaFree(src_d) );
			dst_d = 0;
			src_d = 0;
		}
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float ResizeImage3D_Tricubic(float* dst, const float* src, const int src_width, const int src_height, const float src_depth,
				const int dst_width, const int dst_height, const int dst_depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((dst_width-1)/blockSize.x+1,(dst_height-1)/blockSize.y+1);

		float* dst_d = 0;
		float* src_d = 0;

		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*dst_width*dst_height*dst_depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*src_width*src_height*src_depth*nChannels) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*dst_width*dst_height*dst_depth*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*src_width*src_height*src_depth*nChannels,cudaMemcpyHostToDevice) );

		ResizeImage_Tricubic_Kernel<<<gridSize,blockSize>>>(dst_d,src_d,src_width,src_height,src_depth,dst_width,dst_height,dst_depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*dst_width*dst_height*dst_depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(dst_d) );
		checkCudaErrors( cudaFree(src_d) );
		dst_d = 0;
		src_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float Addwith3D(float* in_out_put, const float* other, const float weight, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* in_out_put_d = 0;
		float* other_d = 0;

		checkCudaErrors( cudaMalloc((void**)&in_out_put_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&other_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(in_out_put_d,in_out_put,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(other_d,other,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );

		Addwith_Kernel<<<gridSize,blockSize>>>(in_out_put_d,other_d,weight,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(in_out_put,in_out_put_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(in_out_put_d) );
		checkCudaErrors( cudaFree(other_d) );
		in_out_put_d = 0;
		other_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float Add3D_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2,
									const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* ouput_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&ouput_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(ouput_d,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );

		Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(ouput_d,Im1_d,weight1,Im2_d,weight2,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(output,ouput_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(ouput_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		ouput_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float GausssianSmoothing3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* src_d = 0;
		float* dst_d = 0;
		float* tmp_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&tmp_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(tmp_d,0,sizeof(float)*width*height*depth*nChannels) );

		Imfilter_h_Gaussian_Kernel<<<gridSize,blockSize>>>(tmp_d,src_d,width,height,depth,nChannels);
		Imfilter_v_Gaussian_Kernel<<<gridSize,blockSize>>>(dst_d,tmp_d,width,height,depth,nChannels);
		Imfilter_d_Gaussian_Kernel<<<gridSize,blockSize>>>(tmp_d,dst_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,tmp_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(src_d) );
		checkCudaErrors( cudaFree(dst_d) );
		checkCudaErrors( cudaFree(tmp_d) );
		src_d = 0;
		dst_d = 0;
		tmp_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	extern "C"
	float GaussianSmoothing2_3D(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);
		
		float* filter = new float[2*fsize+1];
		float sum = 0;
		float m_sigma = sigma*sigma*2;
		for(int i = -fsize;i <= fsize;i++)
		{
			filter[i+fsize] = exp(-(float)(i*i)/m_sigma);
			sum += filter[i+fsize];
		}
		for(int i = 0;i < 2*fsize+1;i++)
			filter[i] /= sum;

		
		float* src_d = 0;
		float* dst_d = 0;
		float* tmp_d = 0;
		float* filter_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&tmp_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&filter_d,sizeof(float)*(2*fsize+1)) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(tmp_d,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(filter_d,filter,sizeof(float)*(2*fsize+1),cudaMemcpyHostToDevice) );

		Imfilter_h_Kernel<<<gridSize,blockSize>>>(tmp_d,src_d,fsize,filter_d,width,height,depth,nChannels);
		Imfilter_v_Kernel<<<gridSize,blockSize>>>(dst_d,tmp_d,fsize,filter_d,width,height,depth,nChannels);
		Imfilter_d_Kernel<<<gridSize,blockSize>>>(tmp_d,dst_d,fsize,filter_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,tmp_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(src_d) );
		checkCudaErrors( cudaFree(dst_d) );
		checkCudaErrors( cudaFree(tmp_d) );
		checkCudaErrors( cudaFree(filter_d) );
		delete []filter;
		src_d = 0;
		dst_d = 0;
		tmp_d = 0;
		filter_d = 0;
		filter = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	extern "C"
	float DerivativeX3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*depth*nChannels) );

		Derivative_x_Advanced_Kernel<<<gridSize,blockSize>>>(dst_d,src_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(src_d) );
		checkCudaErrors( cudaFree(dst_d) );
		src_d = 0;
		dst_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	extern "C"
	float DerivativeY3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*depth*nChannels) );

		Derivative_y_Advanced_Kernel<<<gridSize,blockSize>>>(dst_d,src_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(src_d) );
		checkCudaErrors( cudaFree(dst_d) );
		src_d = 0;
		dst_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float DerivativeZ3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*depth*nChannels) );

		Derivative_z_Advanced_Kernel<<<gridSize,blockSize>>>(dst_d,src_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(src_d) );
		checkCudaErrors( cudaFree(dst_d) );
		src_d = 0;
		dst_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float Laplacian3D(float* dst, const float* src, const int width, const int height, const int depth, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);

		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*depth*nChannels) );

		Laplacian_Kernel<<<gridSize,blockSize>>>(dst_d,src_d,width,height,depth,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(src_d) );
		checkCudaErrors( cudaFree(dst_d) );
		src_d = 0;
		dst_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
}

#endif