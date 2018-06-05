#ifndef _ZQ_CUDA_IMAGE_PROCESSING_2D_CU_
#define _ZQ_CUDA_IMAGE_PROCESSING_2D_CU_

#include "ZQ_CUDA_ImageProcessing2D.cuh"

namespace ZQ_CUDA_ImageProcessing2D
{
	/***************************************************************************/
	
	__global__ void WarpImage_Bilinear_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels)
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

		float target_x = x + u[offset];
		float target_y = y + v[offset];

		if (target_x < 0 || target_x > width - 1 || target_y < 0 || target_y > height - 1)
		{
			for (int c = 0; c < nChannels; c++)
				warpIm2[offset*nChannels + c] = Im1[offset*nChannels + c];
		}
		else
		{
			//float coord_x = (target_x + 0.5f) / width;
			//float coord_y = (target_y + 0.5f) / height;
			float coord_x = target_x;
			float coord_y = target_y;

			for (int c = 0; c < nChannels; c++)
			{
				int x0 = floor(coord_x);
				int x1 = x0 + 1;
				int y0 = floor(coord_y);
				int y1 = y0 + 1;

				float sx = coord_x - x0;
				float sy = coord_y - y0;

				float val = 0.0f;
				int real_x0 = clamp(x0, 0, width - 1);
				int real_x1 = clamp(x1, 0, width - 1);
				int real_y0 = clamp(y0, 0, height - 1);
				int real_y1 = clamp(y1, 0, height - 1);
				val += Im2[(real_y0*width + real_x0)*nChannels + c] * (1.0f - sx)*(1.0f - sy);
				val += Im2[(real_y1*width + real_x0)*nChannels + c] * (1.0f - sx)*      sy;
				val += Im2[(real_y0*width + real_x1)*nChannels + c] * sx *(1.0f - sy);
				val += Im2[(real_y1*width + real_x1)*nChannels + c] * sx *      sy;
				warpIm2[offset*nChannels + c] = val;
			}
		}
	}

	__global__ void WarpImage_Bilinear_Occupy_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels)
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

		float target_x = x + u[offset];
		float target_y = y + v[offset];

		if (occupy[y*width + x] > 0.5 || target_x < 0 || target_x > width - 1 || target_y < 0 || target_y > height - 1 || occupy[(int)target_y*width + (int)target_x] > 0.5)
		{
			for (int c = 0; c < nChannels; c++)
				warpIm2[offset*nChannels + c] = Im1[offset*nChannels + c];
		}
		else
		{
			//float coord_x = (target_x + 0.5f) / width;
			//float coord_y = (target_y + 0.5f) / height;
			float coord_x = target_x;
			float coord_y = target_y;

			for (int c = 0; c < nChannels; c++)
			{
				int x0 = floor(coord_x);
				int x1 = x0 + 1;
				int y0 = floor(coord_y);
				int y1 = y0 + 1;

				float sx = coord_x - x0;
				float sy = coord_y - y0;

				float val = 0.0f;
				int real_x0 = clamp(x0, 0, width - 1);
				int real_x1 = clamp(x1, 0, width - 1);
				int real_y0 = clamp(y0, 0, height - 1);
				int real_y1 = clamp(y1, 0, height - 1);
				val += Im2[(real_y0*width + real_x0)*nChannels + c] * (1.0f - sx)*(1.0f - sy);
				val += Im2[(real_y1*width + real_x0)*nChannels + c] * (1.0f - sx)*      sy;
				val += Im2[(real_y0*width + real_x1)*nChannels + c] * sx *(1.0f - sy);
				val += Im2[(real_y1*width + real_x1)*nChannels + c] * sx *      sy;
				warpIm2[offset*nChannels + c] = val;
			}
		}
	}

	__global__ void WarpImage_Bicubic_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels)
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

		float target_x = x + u[offset];
		float target_y = y + v[offset];

		if (target_x < 0 || target_x > width - 1 || target_y < 0 || target_y > height - 1)
		{
			for (int c = 0; c < nChannels; c++)
				warpIm2[offset*nChannels + c] = Im1[offset*nChannels + c];
		}
		else
		{
			float coord_x = target_x;
			float coord_y = target_y;

			int ix = floor(coord_x);
			int iy = floor(coord_y);
			float fx = coord_x - ix;
			float fy = coord_y - iy;

			for (int c = 0; c < nChannels; c++)
			{
				float data_y[4];

				float dk, dk1, deltak, a3, a2, a1, a0;

				for (int i = 0; i < 4; i++)
				{
					float data_x[4];
					int tmp_y = clamp(iy - 1 + i, 0, height - 1);
					for (int j = 0; j < 4; j++)
					{
						int tmp_x = clamp(ix - 1 + j, 0, width - 1);
						data_x[j] = Im2[(tmp_y*width + tmp_x)*nChannels + c];
					}

					// bicubic interpolation for dimension y
					dk = 0.5*(data_x[2] - data_x[0]);
					dk1 = 0.5*(data_x[3] - data_x[1]);
					deltak = data_x[2] - data_x[1];

					if (deltak == 0)
						dk = dk1 = 0;
					else
					{
						if (dk*deltak < 0)
							dk = 0;
						if (dk1*deltak < 0)
							dk1 = 0;
					}


					a3 = dk + dk1 - 2 * deltak;
					a2 = 3 * deltak - 2 * dk - dk1;
					a1 = dk;
					a0 = data_x[1];

					data_y[i] = a0 + fx*(a1 + fx*(a2 + fx*a3));

				}

				// bicubic interpolation for dimension x
				dk = 0.5*(data_y[2] - data_y[0]);
				dk1 = 0.5*(data_y[3] - data_y[1]);
				deltak = data_y[2] - data_y[1];


				if (deltak == 0)
					dk = dk1 = 0;
				else
				{
					if (dk*deltak < 0)
						dk = 0;
					if (dk1*deltak < 0)
						dk1 = 0;
				}



				a3 = dk + dk1 - 2 * deltak;
				a2 = 3 * deltak - 2 * dk - dk1;
				a1 = dk;
				a0 = data_y[1];

				warpIm2[offset*nChannels + c] = a0 + fy*(a1 + fy*(a2 + fy*a3));
			}
		}
	}
	__global__ void WarpImage_Bicubic_Occupy_Kernel(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels)
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

		float target_x = x + u[offset];
		float target_y = y + v[offset];

		if (occupy[y*width + x] > 0.5 || target_x < 0 || target_x > width - 1 || target_y < 0 || target_y > height - 1 || occupy[(int)target_y*width + (int)target_x] > 0.5)
		{
			for (int c = 0; c < nChannels; c++)
				warpIm2[offset*nChannels + c] = Im1[offset*nChannels + c];
		}
		else
		{
			float coord_x = target_x;
			float coord_y = target_y;

			int ix = floor(coord_x);
			int iy = floor(coord_y);
			float fx = coord_x - ix;
			float fy = coord_y - iy;

			for (int c = 0; c < nChannels; c++)
			{
				float data_y[4];

				float dk, dk1, deltak, a3, a2, a1, a0;

				for (int i = 0; i < 4; i++)
				{
					float data_x[4];
					int tmp_y = clamp(iy - 1 + i, 0, height - 1);
					for (int j = 0; j < 4; j++)
					{
						int tmp_x = clamp(ix - 1 + j, 0, width - 1);
						data_x[j] = Im2[(tmp_y*width + tmp_x)*nChannels + c];
					}

					// bicubic interpolation for dimension y
					dk = 0.5*(data_x[2] - data_x[0]);
					dk1 = 0.5*(data_x[3] - data_x[1]);
					deltak = data_x[2] - data_x[1];

					if (deltak == 0)
						dk = dk1 = 0;
					else
					{
						if (dk*deltak < 0)
							dk = 0;
						if (dk1*deltak < 0)
							dk1 = 0;
					}


					a3 = dk + dk1 - 2 * deltak;
					a2 = 3 * deltak - 2 * dk - dk1;
					a1 = dk;
					a0 = data_x[1];

					data_y[i] = a0 + fx*(a1 + fx*(a2 + fx*a3));

				}

				// bicubic interpolation for dimension x
				dk = 0.5*(data_y[2] - data_y[0]);
				dk1 = 0.5*(data_y[3] - data_y[1]);
				deltak = data_y[2] - data_y[1];


				if (deltak == 0)
					dk = dk1 = 0;
				else
				{
					if (dk*deltak < 0)
						dk = 0;
					if (dk1*deltak < 0)
						dk1 = 0;
				}



				a3 = dk + dk1 - 2 * deltak;
				a2 = 3 * deltak - 2 * dk - dk1;
				a1 = dk;
				a0 = data_y[1];

				warpIm2[offset*nChannels + c] = a0 + fy*(a1 + fy*(a2 + fy*a3));
			}
		}
	}
	
	__global__ void ResizeImage_Bilinear_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;
		if (x >= dst_width || y >= dst_height)
			return;

		int offset = y*dst_width + x;

		float coord_x = (x + 0.5f) / dst_width*src_width - 0.5f;
		float coord_y = (y + 0.5f) / dst_height*src_height - 0.5f;

		int x0 = floor(coord_x);
		int x1 = x0 + 1;
		int y0 = floor(coord_y);
		int y1 = y0 + 1;

		float sx = coord_x - x0;
		float sy = coord_y - y0;

		int real_x0 = clamp(x0, 0, src_width - 1);
		int real_x1 = clamp(x1, 0, src_width - 1);
		int real_y0 = clamp(y0, 0, src_height - 1);
		int real_y1 = clamp(y1, 0, src_height - 1);

		for (int c = 0; c < nChannels; c++)
		{
			float val = 0;
			val += src[(real_y0*src_width + real_x0)*nChannels + c] * (1.0f - sx)*(1.0f - sy);
			val += src[(real_y0*src_width + real_x1)*nChannels + c] * sx *(1.0f - sy);
			val += src[(real_y1*src_width + real_x0)*nChannels + c] * (1.0f - sx)*      sy;
			val += src[(real_y1*src_width + real_x1)*nChannels + c] * sx *      sy;

			dst[offset*nChannels + c] = val;
		}
	}

	__global__ void ResizeImage_Bicubic_Kernel(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;
		if (x >= dst_width || y >= dst_height)
			return;

		int offset = y*dst_width + x;

		float coord_x = (x + 0.5f) / dst_width*src_width - 0.5f;
		float coord_y = (y + 0.5f) / dst_height*src_height - 0.5f;

		int ix = floor(coord_x);
		int iy = floor(coord_y);
		float fx = coord_x - ix;
		float fy = coord_y - iy;

		for (int c = 0; c < nChannels; c++)
		{
			float data_y[4];

			float dk, dk1, deltak, a3, a2, a1, a0;

			for (int i = 0; i < 4; i++)
			{
				float data_x[4];
				int tmp_y = clamp(iy - 1 + i, 0, src_height - 1);
				for (int j = 0; j < 4; j++)
				{
					int tmp_x = clamp(ix - 1 + j, 0, src_width - 1);
					data_x[j] = src[(tmp_y*src_width + tmp_x)*nChannels + c];
				}

				// bicubic interpolation for dimension y
				dk = 0.5*(data_x[2] - data_x[0]);
				dk1 = 0.5*(data_x[3] - data_x[1]);
				deltak = data_x[2] - data_x[1];

				if (deltak == 0)
					dk = dk1 = 0;
				else
				{
					if (dk*deltak < 0)
						dk = 0;
					if (dk1*deltak < 0)
						dk1 = 0;
				}


				a3 = dk + dk1 - 2 * deltak;
				a2 = 3 * deltak - 2 * dk - dk1;
				a1 = dk;
				a0 = data_x[1];

				data_y[i] = a0 + fx*(a1 + fx*(a2 + fx*a3));

			}

			// bicubic interpolation for dimension x
			dk = 0.5*(data_y[2] - data_y[0]);
			dk1 = 0.5*(data_y[3] - data_y[1]);
			deltak = data_y[2] - data_y[1];


			if (deltak == 0)
				dk = dk1 = 0;
			else
			{
				if (dk*deltak < 0)
					dk = 0;
				if (dk1*deltak < 0)
					dk1 = 0;
			}

			a3 = dk + dk1 - 2 * deltak;
			a2 = 3 * deltak - 2 * dk - dk1;
			a1 = dk;
			a0 = data_y[1];

			dst[offset*nChannels + c] = a0 + fy*(a1 + fy*(a2 + fy*a3));
		}
	}

	__global__ void Addwith_Kernel(float* in_out_put, const float* other, const float weight, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		for (int c = 0; c < nChannels; c++)
		{
			in_out_put[offset*nChannels + c] += other[offset*nChannels + c] * weight;
		}
	}

	__global__ void MulWithScale_Kernel(float* in_out_put, const float scale, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		for (int c = 0; c < nChannels; c++)
		{
			in_out_put[offset*nChannels + c] *= scale;
		}
	}

	__global__ void Add_Im1_weight1_Im2_weight2_Kernel(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2,
		const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		for (int c = 0; c < nChannels; c++)
		{
			output[offset*nChannels + c] = Im1[offset*nChannels + c] * weight1 + Im2[offset*nChannels + c] * weight2;
		}
	}

	__global__ void Imfilter_h_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		float pfilter1D[5] = { 0.02, 0.11, 0.74, 0.11, 0.02 };
		int fsize = 2;

		for (int c = 0; c < nChannels; c++)
			output[offset*nChannels + c] = 0;

		for (int l = -fsize; l <= fsize; l++)
		{
			int xx = clamp(x + l, 0, width - 1);
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] += input[(y*width + xx)*nChannels + c] * pfilter1D[l + fsize];
		}
	}

	__global__ void Imfilter_v_Gaussian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		float pfilter1D[5] = { 0.02, 0.11, 0.74, 0.11, 0.02 };
		int fsize = 2;

		for (int c = 0; c < nChannels; c++)
			output[offset*nChannels + c] = 0;

		for (int l = -fsize; l <= fsize; l++)
		{
			int yy = clamp(y + l, 0, height - 1);
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] += input[(yy*width + x)*nChannels + c] * pfilter1D[l + fsize];
		}
	}

	__global__ void Imfilter_h_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		for (int c = 0; c < nChannels; c++)
			output[offset*nChannels + c] = 0;

		for (int l = -fsize; l <= fsize; l++)
		{
			int xx = clamp(x + l, 0, width - 1);
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] += input[(y*width + xx)*nChannels + c] * filter[l + fsize];
		}
	}

	__global__ void Imfilter_v_Kernel(float* output, const float* input, const int fsize, const float* filter, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		for (int c = 0; c < nChannels; c++)
			output[offset*nChannels + c] = 0;

		for (int l = -fsize; l <= fsize; l++)
		{
			int yy = clamp(y + l, 0, height - 1);
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] += input[(yy*width + x)*nChannels + c] * filter[l + fsize];
		}
	}

	__global__ void Derivative_x_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		float pfilter1D[5] = { 1.0 / 12, -8.0 / 12, 0, 8.0 / 12, -1.0 / 12 };
		int fsize = 2;

		for (int c = 0; c < nChannels; c++)
			output[offset*nChannels + c] = 0;

		for (int l = -fsize; l <= fsize; l++)
		{
			int xx = clamp(x + l, 0, width - 1);
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] += input[(y*width + xx)*nChannels + c] * pfilter1D[l + fsize];
		}
	}


	__global__ void Derivative_y_Advanced_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		float pfilter1D[5] = { 1.0 / 12, -8.0 / 12, 0, 8.0 / 12, -1.0 / 12 };
		int fsize = 2;

		for (int c = 0; c < nChannels; c++)
			output[offset*nChannels + c] = 0;

		for (int l = -fsize; l <= fsize; l++)
		{
			int yy = clamp(y + l, 0, height - 1);
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] += input[(yy*width + x)*nChannels + c] * pfilter1D[l + fsize];
		}
	}

	__global__ void Dx_Forward_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		if (x == width - 1)
		{
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] = 0;
		}
		else
		{
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] = input[(offset + 1)*nChannels + c] - input[offset*nChannels + c];
		}
	}


	__global__ void Dy_Forward_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		if (y == height - 1)
		{
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] = 0;
		}
		else
		{
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] = input[(offset + width)*nChannels + c] - input[offset*nChannels + c];
		}
	}


	__global__ void Laplacian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;

		int offset = y*width + x;

		for (int c = 0; c < nChannels; c++)
		{
			float value = 0;
			if (x == 0)
			{
				value += input[(offset + 1)*nChannels + c] - input[offset*nChannels + c];
			}
			else if (x == width - 1)
			{
				value += input[(offset - 1)*nChannels + c] - input[offset*nChannels + c];
			}
			else
			{
				value += input[(offset + 1)*nChannels + c] + input[(offset - 1)*nChannels + c] - 2 * input[offset*nChannels + c];
			}

			if (y == 0)
			{
				value += input[(offset + width)*nChannels + c] - input[offset*nChannels + c];
			}
			else if (y == height - 1)
			{
				value += input[(offset - width)*nChannels + c] - input[offset*nChannels + c];
			}
			else
			{
				value += input[(offset + width)*nChannels + c] + input[(offset - width)*nChannels + c] - 2 * input[offset*nChannels + c];
			}

			output[offset*nChannels + c] = value;
		}
	}

	__global__ void CopyChannel_i_Kernel(float* output, const float* input, const int i, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;
		int offset = y*width + x;
		output[offset] = input[offset*nChannels + i];
	}

	__global__ void MedianFilterWithMask5x5_Kernel(float* output, const float* input, const int width, const int height, const int nChannels, const bool* keep_mask)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;
		int offset = y*width + x;
		if (keep_mask[offset])
		{
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] = input[offset*nChannels + c];
			return;
		}

		float vals[25] = { 0 };
		int count = 0;
		for (int c = 0; c < nChannels; c++)
		{
			count = 0;
			int start_x = ((x - 2) >= 0) ? (x - 2) : 0;
			int end_x = ((x + 2) <= (width - 1)) ? (x + 2) : (width - 1);
			int start_y = ((y - 2) >= 0) ? (y - 2) : 0;
			int end_y = ((y + 2) <= (height - 1)) ? (y + 2) : (height - 1);
			for (int ii = start_y; ii <= end_y; ii++)
			{
				for (int jj = start_x; jj <= end_x; jj++)
				{
					int cur_offset = ii*width + jj;
					if (keep_mask[cur_offset])
					{
						vals[count++] = input[cur_offset*nChannels + c];
					}
				}
			}
			if (count == 0)
			{
				output[offset*nChannels + c] = 0;
			}
			else
			{
				int mid = (count + 1) / 2;
				for (int pass = 0; pass < mid; pass++)
				{
					float max_val = vals[pass];
					int max_id = pass;
					for (int id = pass + 1; id < count; id++)
					{
						if (max_val < vals[id])
						{
							max_val = vals[id];
							max_id = id;
						}
					}
					vals[max_id] = vals[pass];
					vals[pass] = max_val;
				}
				output[offset*nChannels + c] = vals[mid];
			}
		}
	}

	__global__ void MedianFilterWithMask3x3_Kernel(float* output, const float* input, const int width, const int height, const int nChannels, const bool* keep_mask)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height)
			return;
		int offset = y*width + x;
		if (keep_mask[offset])
		{
			for (int c = 0; c < nChannels; c++)
				output[offset*nChannels + c] = input[offset*nChannels + c];
			return;
		}

		float vals[9] = { 0 };
		int count = 0;
		for (int c = 0; c < nChannels; c++)
		{
			count = 0;
			int start_x = ((x - 1) >= 0) ? (x - 1) : 0;
			int end_x = ((x + 1) <= (width - 1)) ? (x + 1) : (width - 1);
			int start_y = ((y - 1) >= 0) ? (y - 1) : 0;
			int end_y = ((y + 1) <= (height - 1)) ? (y + 1) : (height - 1);
			for (int ii = start_y; ii <= end_y; ii++)
			{
				for (int jj = start_x; jj <= end_x; jj++)
				{
					int cur_offset = ii*width + jj;
					if (keep_mask[cur_offset])
					{
						vals[count++] = input[cur_offset*nChannels + c];
					}
				}
			}
			if (count == 0)
			{
				output[offset*nChannels + c] = 0;
			}
			else
			{
				int mid = (count + 1) / 2;
				for (int pass = 0; pass < mid; pass++)
				{
					float max_val = vals[pass];
					int max_id = pass;
					for (int id = pass + 1; id < count; id++)
					{
						if (max_val < vals[id])
						{
							max_val = vals[id];
							max_id = id;
						}
					}
					vals[max_id] = vals[pass];
					vals[pass] = max_val;
				}
				output[offset*nChannels + c] = vals[mid];
			}
		}
	}
	/******************************************************/

	void cu_WarpImage_Bilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels)\
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		WarpImage_Bilinear_Kernel << <gridSize, blockSize >> >(warpIm2, Im1, Im2, u, v, width, height, nChannels);
	}

	extern "C"
	float WarpImage2D_Bilinear(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* warpIm2_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;

		checkCudaErrors(cudaMalloc((void**)&warpIm2_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&u_d, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&v_d, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&Im1_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&Im2_d, sizeof(float)*width*height*nChannels));

		checkCudaErrors(cudaMemset(warpIm2_d, 0, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemcpy(u_d, u, sizeof(float)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(v_d, v, sizeof(float)*width*height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(Im1_d, Im1, sizeof(float)*width*height * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(Im2_d, Im2, sizeof(float)*width*height * 4, cudaMemcpyHostToDevice));

		cu_WarpImage_Bilinear(warpIm2_d, Im1_d, Im2_d, u_d, v_d, width, height, nChannels);

		checkCudaErrors(cudaMemcpy(warpIm2, warpIm2_d, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(warpIm2_d));
		checkCudaErrors(cudaFree(u_d));
		checkCudaErrors(cudaFree(v_d));
		checkCudaErrors(cudaFree(Im1_d));
		checkCudaErrors(cudaFree(Im2_d));
		warpIm2_d = 0;
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float WarpImage2D_Bilinear_Occupy(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels)
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
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height,cudaMemcpyHostToDevice) );

		WarpImage_Bilinear_Occupy_Kernel<<<gridSize,blockSize>>>(warpIm2_d,Im1_d,Im2_d,occupy_d,u_d,v_d,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		warpIm2_d = 0;
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		occupy_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	void cu_WarpImage_Bicubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		WarpImage_Bicubic_Kernel << <gridSize, blockSize >> >(warpIm2, Im1, Im2, u, v, width, height, nChannels);
	}

	extern "C"
	float WarpImage2D_Bicubic(float* warpIm2, const float* Im1, const float* Im2, const float* u, const float* v, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* warpIm2_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );

		cu_WarpImage_Bicubic(warpIm2_d, Im1_d, Im2_d, u_d, v_d, width, height, nChannels);

		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		warpIm2_d = 0;
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float WarpImage2D_Bicubic_Occupy(float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* u, const float* v, const int width, const int height, const int nChannels)
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
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;

		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(warpIm2_d,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height,cudaMemcpyHostToDevice) );

		WarpImage_Bicubic_Occupy_Kernel<<<gridSize,blockSize>>>(warpIm2_d,Im1_d,Im2_d,occupy_d,u_d,v_d,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		warpIm2_d = 0;
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		occupy_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	void cu_ResizeImage_Bilinear(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((dst_width - 1) / blockSize.x + 1, (dst_height - 1) / blockSize.y + 1);

		ResizeImage_Bilinear_Kernel << <gridSize, blockSize >> >(dst, src, src_width, src_height, dst_width, dst_height, nChannels);
	}
	
	extern "C" float ResizeImage2D_Bilinear(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);


		float* dst_d = 0;
		float* src_d = 0;

		checkCudaErrors(cudaMalloc((void**)&dst_d, sizeof(float)*dst_width*dst_height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&src_d, sizeof(float)*src_width*src_height*nChannels));
		checkCudaErrors(cudaMemset(dst_d, 0, sizeof(float)*dst_width*dst_height*nChannels));
		checkCudaErrors(cudaMemcpy(src_d, src, sizeof(float)*src_width*src_height*nChannels, cudaMemcpyHostToDevice));

		cu_ResizeImage_Bilinear(dst_d, src_d, src_width, src_height, dst_width, dst_height, nChannels);

		checkCudaErrors(cudaMemcpy(dst, dst_d, sizeof(float)*dst_width*dst_height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(dst_d));
		checkCudaErrors(cudaFree(src_d));
		dst_d = 0;
		src_d = 0;


		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

	void cu_ResizeImage_Bicubic(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((dst_width - 1) / blockSize.x + 1, (dst_height - 1) / blockSize.y + 1);

		ResizeImage_Bicubic_Kernel << <gridSize, blockSize >> >(dst, src, src_width, src_height, dst_width, dst_height, nChannels);
	}

	extern "C" float ResizeImage2D_Bicubic(float* dst, const float* src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((dst_width - 1) / blockSize.x + 1, (dst_height - 1) / blockSize.y + 1);

		float* dst_d = 0;
		float* src_d = 0;

		checkCudaErrors(cudaMalloc((void**)&dst_d, sizeof(float)*dst_width*dst_height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&src_d, sizeof(float)*src_width*src_height*nChannels));
		checkCudaErrors(cudaMemset(dst_d, 0, sizeof(float)*dst_width*dst_height*nChannels));
		checkCudaErrors(cudaMemcpy(src_d, src, sizeof(float)*src_width*src_height*nChannels, cudaMemcpyHostToDevice));

		cu_ResizeImage_Bicubic(dst_d, src_d, src_width, src_height, dst_width, dst_height, nChannels);

		checkCudaErrors(cudaMemcpy(dst, dst_d, sizeof(float)*dst_width*dst_height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(dst_d));
		checkCudaErrors(cudaFree(src_d));
		dst_d = 0;
		src_d = 0;

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

	void cu_Addwith(float* in_out_put, const float* other, const float weight, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Addwith_Kernel << <gridSize, blockSize >> >(in_out_put, other, weight, width, height, nChannels);
	}


	extern "C" float Addwith2D(float* in_out_put, const float* other, const float weight, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* in_out_put_d = 0;
		float* other_d = 0;

		checkCudaErrors( cudaMalloc((void**)&in_out_put_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&other_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(in_out_put_d,in_out_put,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(other_d,other,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );

		cu_Addwith(in_out_put_d,other_d,weight,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(in_out_put,in_out_put_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

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

	void cu_MulWithScale(float* in_out_put, const float scale, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		MulWithScale_Kernel << <gridSize, blockSize >> >(in_out_put, scale, width, height, nChannels);
	}

	extern "C" float MulWithScale2D(float* in_out_put, const float scale, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		float* in_out_put_d = 0;
	
		checkCudaErrors(cudaMalloc((void**)&in_out_put_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemcpy(in_out_put_d, in_out_put, sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice));

		cu_MulWithScale(in_out_put_d, scale, width, height, nChannels);

		checkCudaErrors(cudaMemcpy(in_out_put, in_out_put_d, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(in_out_put_d));
		in_out_put_d = 0;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

	void cu_Add_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2,
		const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Add_Im1_weight1_Im2_weight2_Kernel << <gridSize, blockSize >> >(output, Im1, weight1, Im2, weight2, width, height, nChannels);
	}

	extern "C" float Add2D_Im1_weight1_Im2_weight2(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2,
									const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* ouput_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&ouput_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(ouput_d,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );

		cu_Add_Im1_weight1_Im2_weight2(ouput_d,Im1_d,weight1,Im2_d,weight2,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(output,ouput_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

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

	void cu_GaussianSmoothing(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		float* tmp = 0;
		checkCudaErrors(cudaMalloc((void**)&tmp, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemset(tmp, 0, sizeof(float)*width*height*nChannels));

		Imfilter_h_Gaussian_Kernel << <gridSize, blockSize >> >(tmp, src, width, height, nChannels);
		Imfilter_v_Gaussian_Kernel << <gridSize, blockSize >> >(dst, tmp, width, height, nChannels);

		checkCudaErrors(cudaFree(tmp));
		tmp = 0;
	}

	extern "C" float GaussianSmoothing2D(float* dst, const float* src, const int width, const int height, const int nChannels)
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

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*nChannels) );

		cu_GaussianSmoothing(dst_d, src_d, width, height, nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

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
	
	void cu_GaussianSmoothing2(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		float* filter = new float[2 * fsize + 1];
		float sum = 0;
		float m_sigma = sigma*sigma * 2;
		for (int i = -fsize; i <= fsize; i++)
		{
			filter[i + fsize] = exp(-(float)(i*i) / m_sigma);
			sum += filter[i + fsize];
		}
		for (int i = 0; i < 2 * fsize + 1; i++)
			filter[i] /= sum;

		float* filter_d = 0;
		checkCudaErrors(cudaMalloc((void**)&filter_d, sizeof(float)*(2 * fsize + 1)));
		checkCudaErrors(cudaMemcpy(filter_d, filter, sizeof(float)*(2 * fsize + 1), cudaMemcpyHostToDevice));
		float* tmp = 0;
		checkCudaErrors(cudaMalloc((void**)&tmp, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemset(tmp, 0, sizeof(float)*width*height*nChannels));

		Imfilter_h_Kernel << <gridSize, blockSize >> >(tmp, src, fsize, filter_d, width, height, nChannels);
		Imfilter_v_Kernel << <gridSize, blockSize >> >(dst, tmp, fsize, filter_d, width, height, nChannels);

		checkCudaErrors(cudaFree(filter_d));
		checkCudaErrors(cudaFree(tmp));
		delete[]filter;
		filter_d = 0;
		filter = 0;
	}


	extern "C"
	float GaussianSmoothing2_2D(float* dst, const float* src, const float sigma, const int fsize, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*nChannels) );

		cu_GaussianSmoothing2(dst_d, src_d, sigma, fsize, width, height, nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

		checkCudaErrors(cudaFree(src_d));
		checkCudaErrors(cudaFree(dst_d));
		src_d = 0;
		dst_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}


	void cu_DerivativeX(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Derivative_x_Advanced_Kernel << <gridSize, blockSize >> >(dst, src, width, height, nChannels);
	}

	extern "C"
	float DerivativeX2D(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*nChannels) );

		cu_DerivativeX(dst_d,src_d,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

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

	void cu_DerivativeY(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Derivative_y_Advanced_Kernel << <gridSize, blockSize >> >(dst, src, width, height, nChannels);
	}

	extern "C"
	float DerivativeY2D(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*nChannels) );

		cu_DerivativeY(dst_d,src_d,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

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

	void cu_DxForward(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Dx_Forward_Kernel << <gridSize, blockSize >> >(dst, src, width, height, nChannels);
	}

	extern "C" float DxForward2D(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors(cudaMalloc((void**)&src_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&dst_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemcpy(src_d, src, sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(dst_d, 0, sizeof(float)*width*height*nChannels));

		cu_DxForward(dst_d, src_d, width, height, nChannels);

		checkCudaErrors(cudaMemcpy(dst, dst_d, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(src_d));
		checkCudaErrors(cudaFree(dst_d));
		src_d = 0;
		dst_d = 0;

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

	void cu_DyForward(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Dy_Forward_Kernel << <gridSize, blockSize >> >(dst, src, width, height, nChannels);
	}

	extern "C" float DyForward2D(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors(cudaMalloc((void**)&src_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&dst_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemcpy(src_d, src, sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(dst_d, 0, sizeof(float)*width*height*nChannels));

		cu_DyForward(dst_d, src_d, width, height, nChannels);

		checkCudaErrors(cudaMemcpy(dst, dst_d, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(src_d));
		checkCudaErrors(cudaFree(dst_d));
		src_d = 0;
		dst_d = 0;

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

	void cu_Laplacian(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Laplacian_Kernel << <gridSize, blockSize >> >(dst, src, width, height, nChannels);
	}

	extern "C"
	float Laplacian2D(float* dst, const float* src, const int width, const int height, const int nChannels)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* src_d = 0;
		float* dst_d = 0;

		checkCudaErrors( cudaMalloc((void**)&src_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&dst_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(src_d,src,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemset(dst_d,0,sizeof(float)*width*height*nChannels) );

		cu_Laplacian(dst_d,src_d,width,height,nChannels);

		checkCudaErrors( cudaMemcpy(dst,dst_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );

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

	void cu_CopyChannel_i(float* dst, const float* src, const int i, const int width, const int height, const int nChannels)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		CopyChannel_i_Kernel << <gridSize, blockSize >> >(dst, src, i, width, height, nChannels);
	}

}

#endif