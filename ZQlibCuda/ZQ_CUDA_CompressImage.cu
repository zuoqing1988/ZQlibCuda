#ifndef _ZQ_CUDA_COMPRESS_IMAGE_CU_
#define _ZQ_CUDA_COMPRESS_IMAGE_CU_

#include "ZQ_CUDA_BitonicSort.cuh"
#include "ZQ_CUDA_BaseUtils.cuh"
#include "ZQ_CUDA_Wavelet.cuh"


namespace ZQ_CUDA_CompressImage
{
	__global__
	void Find_Thresh_kernel(const int num, const float* vals, const float thresh_total_energy, float* result)
	{
		int idx = threadIdx.x;
		int STEP_SIZE = blockDim.x;
		
		int slice = (num+STEP_SIZE-1)/STEP_SIZE;

		__shared__ float s_sum[BLOCK_SIZE*BLOCK_SIZE];
		__shared__ float s_off[BLOCK_SIZE*BLOCK_SIZE];
		s_sum[idx] = 0;
		
		for(int i = slice*idx;i < num && i < slice*(idx+1);i++)
		{
			s_sum[idx] += vals[i];
		}
		
		__syncthreads();
		
		if(idx == 0)
		{	
			float total_sum = 0;
			for(int i = 0;i < STEP_SIZE;i++)
			{
				total_sum += s_sum[i];
				s_off[i] = total_sum;
			}	
		
			
			int start_idx = 0;
			
			for(int i = 0;i < STEP_SIZE;i++,start_idx++)
			{
				if(s_off[i] >= thresh_total_energy)
					break;
			}

			if(start_idx == STEP_SIZE)
				result[0] = 0;

			float tmp_sum = 0;

			if(start_idx == 0)
				tmp_sum = 0;
			else
				tmp_sum = s_off[start_idx-1];
			
			for(int i = start_idx*slice;i < num && i < (start_idx+1)*slice;i++)
			{
				result[0] = vals[i];
				tmp_sum += vals[i];
				if(tmp_sum >= thresh_total_energy)
					break;
			}
			
		}
	}

	extern "C"
	float ComputeCoeffsAndThresh(const float* input, const int width, const int height, const int nLevels, const float quality, float*& output,int& output_width, int& output_height, float& thresh)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float time1,time2,time3;
		
		cudaEvent_t start1,stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1,0);

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

		int output_size = output_width*output_height;

		int coeff_size = 1;
		while(coeff_size < output_size)
		{
			coeff_size *= 2;
		}

		checkCudaErrors( cudaMalloc((void**)&output_d,sizeof(float)*coeff_size) );
		checkCudaErrors( cudaMemset(output_d,0,sizeof(float)*coeff_size) );
		
		cudaEventRecord(stop1,0);
		cudaEventSynchronize(start1);
		cudaEventSynchronize(stop1);
		cudaEventElapsedTime(&time1,start1,stop1);
		
		printf("copy in cost:%f\n",0.001*time1);
		
		cudaEvent_t start2,stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2,0);

		ZQ_CUDA_Wavelet::cu_DWT2_NLevels(input_d,padding_width,padding_height,nLevels,output_d);

		checkCudaErrors( cudaMemcpy(output,output_d,sizeof(float)*output_size,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(input_d) );
		
		cudaEventRecord(stop2,0);
		cudaEventSynchronize(start2);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&time2,start2,stop2);
		
		printf("DWT cost:%f\n",0.001*time2);
		
		cudaEvent_t start3,stop3;
		cudaEventCreate(&start3);
		cudaEventCreate(&stop3);
		cudaEventRecord(start3,0);

		ZQ_CUDA_BaseUtils::cu_Square(coeff_size,output_d,output_d);
		ZQ_CUDA_BitonicSort::cu_BitonicSort(output_d,coeff_size,false);

		float sum_energy = 0;
		ZQ_CUDA_BaseUtils::cu_SUM(coeff_size,output_d,sum_energy);


		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		float* thresh_d = 0;
		checkCudaErrors( cudaMalloc((void**)&thresh_d,sizeof(float)*1) );
		checkCudaErrors( cudaMemset(thresh_d,0,sizeof(float)*1) );
		Find_Thresh_kernel<<<gridSize,blockSize>>>(coeff_size,output_d,sum_energy*quality, thresh_d);
		
		checkCudaErrors( cudaMemcpy(&thresh,thresh_d,sizeof(float)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree( thresh_d) );
		
		checkCudaErrors( cudaFree(output_d) );
		
		cudaEventRecord(stop3,0);
		cudaEventSynchronize(start3);
		cudaEventSynchronize(stop3);
		cudaEventElapsedTime(&time3,start3,stop3);
		
		printf("find thresh cost:%f\n",0.001*time3);
				
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float ComputeCoeffsAndThresh3D(const float* input, const int width, const int height, const int depth, const int nLevels, const float quality, float*& output,int& output_width, int& output_height, float& thresh)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float time1,time2,time3;
		
		cudaEvent_t start1,stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1,0);

		int padding_size = pow(2.0f,(float)nLevels);
		
		float* input_d = 0;
		float* output_d = 0;
		int padding_width, padding_height;
		int padding_slice;

		if(width%padding_size == 0 && height%padding_size == 0)
		{
			padding_width = width;
			padding_height = height;
			padding_slice = padding_width*padding_height;
			output_width = padding_width;
			output_height = padding_height;
				
			checkCudaErrors( cudaMalloc((void**)&input_d,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemcpy(input_d,input,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		}
		else
		{
			padding_width = (width+padding_size-1)/padding_size*padding_size;
			padding_height = (height+padding_size-1)/padding_size*padding_size;
			padding_slice = padding_width*padding_height;
			
			output_width = padding_width;
			output_height = padding_height;
			
			checkCudaErrors( cudaMalloc((void**)&input_d,sizeof(float)*padding_width*padding_height*depth) );
			checkCudaErrors( cudaMemset(input_d,0,sizeof(float)*padding_width*padding_height*depth) );
			
			
			float* tmp_input = new float[padding_width*padding_height*depth];
			memset(tmp_input,0,sizeof(float)*padding_width*padding_height*depth);
			for(int k = 0;k < depth;k++)
			{
				for(int i = 0;i < height;i++)
					memcpy(tmp_input+k*padding_slice+i*padding_width,input+k*height*width+i*width,sizeof(float)*width);
			}
			checkCudaErrors( cudaMemcpy(input_d,tmp_input,sizeof(float)*padding_slice*depth, cudaMemcpyHostToDevice) );
			
			delete []tmp_input;

		}

		output = new float[output_width*output_height*depth];
		memset(output,0,sizeof(float)*output_width*output_height*depth);

		int output_size = output_width*output_height*depth;

		int coeff_size = 1;
		while(coeff_size < output_size)
		{
			coeff_size *= 2;
		}

		checkCudaErrors( cudaMalloc((void**)&output_d,sizeof(float)*coeff_size) );
		checkCudaErrors( cudaMemset(output_d,0,sizeof(float)*coeff_size) );
		
		cudaEventRecord(stop1,0);
		cudaEventSynchronize(start1);
		cudaEventSynchronize(stop1);
		cudaEventElapsedTime(&time1,start1,stop1);
		
		printf("copy in cost:%f\n",0.001*time1);
		
		cudaEvent_t start2,stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2,0);

		/*for(int k = 0;k < depth;k++)
		{
			ZQ_CUDA_Wavelet::cu_DWT2_NLevels(input_d+k*padding_slice,padding_width,padding_height,nLevels,output_d+k*padding_slice);
		}*/
		
		ZQ_CUDA_Wavelet::cu_DWT2_NLevels(input_d,padding_width,padding_height,depth,nLevels,output_d);

		checkCudaErrors( cudaMemcpy(output,output_d,sizeof(float)*output_size,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(input_d) );
		
		cudaEventRecord(stop2,0);
		cudaEventSynchronize(start2);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&time2,start2,stop2);
		
		printf("DWT cost:%f\n",0.001*time2);
		
		cudaEvent_t start3,stop3;
		cudaEventCreate(&start3);
		cudaEventCreate(&stop3);
		cudaEventRecord(start3,0);

		ZQ_CUDA_BaseUtils::cu_Square(coeff_size,output_d,output_d);
		ZQ_CUDA_BitonicSort::cu_BitonicSort(output_d,coeff_size,false);

		float sum_energy = 0;
		ZQ_CUDA_BaseUtils::cu_SUM(coeff_size,output_d,sum_energy);


		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		float* thresh_d = 0;
		checkCudaErrors( cudaMalloc((void**)&thresh_d,sizeof(float)*1) );
		checkCudaErrors( cudaMemset(thresh_d,0,sizeof(float)*1) );
		
		float thresh_total_energy = sum_energy*quality;
		
		Find_Thresh_kernel<<<gridSize,blockSize>>>(coeff_size,output_d,thresh_total_energy, thresh_d);
		
		//printf("thresh_total_energy = %f\n",thresh_total_energy);
		
		checkCudaErrors( cudaMemcpy(&thresh,thresh_d,sizeof(float)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree( thresh_d) );
		
		checkCudaErrors( cudaFree(output_d) );
		
		cudaEventRecord(stop3,0);
		cudaEventSynchronize(start3);
		cudaEventSynchronize(stop3);
		cudaEventElapsedTime(&time3,start3,stop3);
		
		printf("find thresh cost:%f\n",0.001*time3);
				
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		
		return time;
	}
}

#endif