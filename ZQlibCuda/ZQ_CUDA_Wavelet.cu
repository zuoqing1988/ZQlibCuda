#ifndef _ZQ_CUDA_WAVELET_CU_
#define _ZQ_CUDA_WAVELET_CU_

#include "ZQ_CUDA_Wavelet.cuh"

namespace ZQ_CUDA_Wavelet
{
	__global__
	void DWT2_horizontal_kernel(const float* input, int global_width, int width, int height, float* output)
	{
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;
		
		int half_width = width/2;
		if(x >= half_width || y >= height)
			return;


		const float l_filter[2] = {0.70710678118654f, 0.70710678118654f};
		const float h_filter[2] = {-0.70710678118654f, 0.70710678118654f};
		const int filter_len = 2;


		float l_result = 0,h_result = 0;
		
		#pragma unroll 2
		for(int i = 0;i < filter_len;i++)
		{
			l_result += input[y*global_width+x*2+i] * l_filter[filter_len-1-i];
			h_result += input[y*global_width+x*2+i] * h_filter[filter_len-1-i];
		}

		output[y*global_width+x] = l_result;
		output[y*global_width+x+half_width] = h_result;
	}

	__global__
	void DWT2_vertical_kernel(const float* input, int global_width, int width, int height, float* output)
	{
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;

		int half_width = width/2;
		int half_height = height/2;
		if(x >= half_width || y >= half_height)
			return;

		const float l_filter[2] = {0.70710678118654f, 0.70710678118654f};
		const float h_filter[2] = {-0.70710678118654f, 0.70710678118654f};
		const int filter_len = 2;

		float ca_result = 0, ch_result = 0;
		float cv_result = 0, cd_result = 0;

		#pragma unroll 2
		for(int i = 0;i < filter_len;i++)
		{
			ca_result += input[(2*y+i)*global_width+x] * l_filter[filter_len-1-i];
			ch_result += input[(2*y+i)*global_width+x] * h_filter[filter_len-1-i];
			cv_result += input[(2*y+i)*global_width+half_width+x] * l_filter[filter_len-1-i];
			cd_result += input[(2*y+i)*global_width+half_width+x] * h_filter[filter_len-1-i];
		}

		output[y*global_width+x] = ca_result;
		output[y*global_width+x+half_width] = ch_result;
		output[(y+half_height)*global_width+x] = cv_result;
		output[(y+half_height)*global_width+x+half_width] = cd_result;
	}

	__global__
	void DWT2_horizontal_kernel(const float* input, int global_slice, int global_width, int width, int height, int depth, float* output)
	{
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;
		
		int half_width = width/2;
		if(x >= half_width || y >= height)
			return;


		const float l_filter[2] = {0.70710678118654f, 0.70710678118654f};
		const float h_filter[2] = {-0.70710678118654f, 0.70710678118654f};
		const int filter_len = 2;


		for(int z = 0;z < depth;z++)
		{
			float l_result = 0,h_result = 0;
		
			#pragma unroll 2
			for(int i = 0;i < filter_len;i++)
			{
				l_result += input[z*global_slice+y*global_width+x*2+i] * l_filter[filter_len-1-i];
				h_result += input[z*global_slice+y*global_width+x*2+i] * h_filter[filter_len-1-i];
			}

			output[z*global_slice+y*global_width+x] = l_result;
			output[z*global_slice+y*global_width+x+half_width] = h_result;
		}
	}

	__global__
	void DWT2_vertical_kernel(const float* input, int global_slice, int global_width, int width, int height, int depth, float* output)
	{
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;

		int half_width = width/2;
		int half_height = height/2;
		if(x >= half_width || y >= half_height)
			return;

		const float l_filter[2] = {0.70710678118654f, 0.70710678118654f};
		const float h_filter[2] = {-0.70710678118654f, 0.70710678118654f};
		const int filter_len = 2;

		for(int z = 0;z < depth; z++)
		{
			float ca_result = 0, ch_result = 0;
			float cv_result = 0, cd_result = 0;

			#pragma unroll 2
			for(int i = 0;i < filter_len;i++)
			{
				ca_result += input[z*global_slice+(2*y+i)*global_width+x] * l_filter[filter_len-1-i];
				ch_result += input[z*global_slice+(2*y+i)*global_width+x] * h_filter[filter_len-1-i];
				cv_result += input[z*global_slice+(2*y+i)*global_width+half_width+x] * l_filter[filter_len-1-i];
				cd_result += input[z*global_slice+(2*y+i)*global_width+half_width+x] * h_filter[filter_len-1-i];
			}

			output[z*global_slice+y*global_width+x] = ca_result;
			output[z*global_slice+y*global_width+x+half_width] = ch_result;
			output[z*global_slice+(y+half_height)*global_width+x] = cv_result;
			output[z*global_slice+(y+half_height)*global_width+x+half_width] = cd_result;
		}
	}
	
	void cu_DWT2_OneLevel(const float* input, const int global_width, const int width, const int height, float* output)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		float* tmp = 0;
		checkCudaErrors( cudaMalloc((void**)&tmp,sizeof(float)*global_width*height) );
		checkCudaErrors( cudaMemset(tmp,0,sizeof(float)*global_width*height) );

		DWT2_horizontal_kernel<<<gridSize,blockSize>>>(input,global_width,width,height,tmp);
		DWT2_vertical_kernel<<<gridSize,blockSize>>>(tmp,global_width,width,height,output);
	}
	
	void cu_DWT2_OneLevel(const float* input,const int global_slice, const int global_width, const int width, const int height, const int depth, float* output)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		float* tmp = 0;
		checkCudaErrors( cudaMalloc((void**)&tmp,sizeof(float)*global_slice*depth) );
		checkCudaErrors( cudaMemset(tmp,0,sizeof(float)*global_slice*depth) );

		DWT2_horizontal_kernel<<<gridSize,blockSize>>>(input,global_slice,global_width,width,height,depth,tmp);
		DWT2_vertical_kernel<<<gridSize,blockSize>>>(tmp,global_slice,global_width,width,height,depth,output);
	}

	void cu_DWT2_NLevels(const float* input, const int width, const int height, const int levels, float* output)
	{
		cu_DWT2_OneLevel(input,width,width,height,output);
		
		int cur_width = width;
		int cur_height = height;
		int cur_level = 1;
		for(; cur_level < levels;cur_level++)
		{
			cur_width /= 2;
			cur_height /= 2;
			
			cu_DWT2_OneLevel(output,width,cur_width,cur_height,output);
			
		}
	}
	
	void cu_DWT2_NLevels(const float* input, const int width, const int height, const int depth, const int levels, float* output)
	{
		cu_DWT2_OneLevel(input,width*height,width,width,height,depth,output);
		
		int cur_width = width;
		int cur_height = height;
		int cur_level = 1;
		for(; cur_level < levels;cur_level++)
		{
			cur_width /= 2;
			cur_height /= 2;
			
			cu_DWT2_OneLevel(output,width*height,width,cur_width,cur_height,depth,output);
			
		}
	}

	/*db1 filter, zero padding mode*/
	extern "C"
	float DWT2(const float* input, const int width, const int height, const int nLevels, float*& output, int& output_width, int& output_height)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		int padding_size = pow(2.0f,(float)nLevels);
		
		float* input_d = 0;
		float* output_d = 0;
		int padding_width, padding_height;

		if(width%padding_size == 0 && height%padding_size == 0)
		{
			padding_width = width;
			padding_height = height;
			output_width = padding_width;
			output_height = padding_height;
				
			checkCudaErrors( cudaMalloc((void**)&input_d,sizeof(float)*width*height) );
			checkCudaErrors( cudaMemcpy(input_d,input,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		}
		else
		{
			padding_width = (width+padding_size-1)/padding_size*padding_size;
			padding_height = (height+padding_size-1)/padding_size*padding_size;

			output_width = padding_width;
			output_height = padding_height;

			checkCudaErrors( cudaMalloc((void**)&input_d,sizeof(float)*padding_width*padding_height) );
			checkCudaErrors( cudaMemset(input_d,0,sizeof(float)*padding_width*padding_height) );
			
			for(int i = 0;i < height;i++)
				checkCudaErrors( cudaMemcpy(input_d+i*padding_width,input+i*width,sizeof(float)*width,cudaMemcpyHostToDevice) );		
		}

		output = new float[output_width*output_height];
		memset(output,0,sizeof(float)*output_width*output_height);

		checkCudaErrors( cudaMalloc((void**)&output_d,sizeof(float)*padding_width*padding_height) );
		checkCudaErrors( cudaMemset(output_d,0,sizeof(float)*padding_width*padding_height) );

		cu_DWT2_NLevels(input_d,padding_width,padding_height,nLevels,output_d);

		checkCudaErrors( cudaMemcpy(output,output_d,sizeof(float)*output_width*output_height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(output_d) );
		checkCudaErrors( cudaFree(input_d) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
}

#endif