#ifndef _ZQ_CUDA_BITONIC_SORT_CU_
#define _ZQ_CUDA_BITONIC_SORT_CU_

#include "ZQ_CUDA_BitonicSort.cuh"

namespace ZQ_CUDA_BitonicSort
{
	__device__
	void BitonicCompare(float& val1, float& val2, bool ascending_dir)
	{
		if (ascending_dir == (val1>val2))
		{
			float tmp = val1;
			val1 = val2;
			val2 = tmp;
		}
	}

	__global__
	void BitonicMerge_kernel(float* data, int len, int sort_block_size, int merge_block_size, int merge_step_size, bool ascending_dir)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx*2 > len)
			return;

		int cur_sort_block_idx = idx/(sort_block_size/2);
		int cur_dir = (cur_sort_block_idx%2 == 0) ? ascending_dir : (!ascending_dir);

		int cur_merge_block_idx = idx/(merge_block_size/2);
		int cur_merge_block_off = idx%(merge_block_size/2);

		int real_i = cur_merge_block_idx*merge_block_size+cur_merge_block_off;
		int real_j = real_i+merge_step_size;
		BitonicCompare(data[real_i],data[real_j],cur_dir);


	}

	void cu_BitonicMerge(float* data, int len, int sort_lvl, int merge_lvl, bool ascending_dir)
	{
		int sort_block_size = 1;
		for(int ll = 0;ll < sort_lvl;ll++)
			sort_block_size *= 2;

		int merge_block_size = 1;
		for(int ll = 0;ll < merge_lvl;ll++)
			merge_block_size *= 2;

		int merge_step_size = merge_block_size/2;

		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((len/2+blockSize.x-1)/blockSize.x,1);

		BitonicMerge_kernel<<<gridSize,blockSize>>>(data,len,sort_block_size,merge_block_size,merge_step_size,ascending_dir);
	}

	bool cu_BitonicSort(float* data, int len, bool ascending_dir)
	{
		int max_levels = 0;
		int cur_len = len;
		while(cur_len > 1)
		{
			if(cur_len%2 != 0)
				return false;
			max_levels ++;
			cur_len /= 2;
		}

		for(int sort_lvl = 1; sort_lvl <= max_levels;sort_lvl++)
		{
			for(int merge_lvl = sort_lvl;merge_lvl > 0;merge_lvl--)
				cu_BitonicMerge(data,len,sort_lvl,merge_lvl,ascending_dir);
		}
		return true;
	}

	/****************************************/
	extern "C"
	bool BitonicSort(float* data, int len, bool ascending_dir, float& cost_time)
	{

		cost_time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* data_d = 0;
		checkCudaErrors( cudaMalloc((void**)&data_d,sizeof(float)*len) );
		checkCudaErrors( cudaMemcpy(data_d,data,sizeof(float)*len,cudaMemcpyHostToDevice) );
		
		bool flag = cu_BitonicSort(data_d,len,ascending_dir);
		
		checkCudaErrors( cudaMemcpy(data,data_d,sizeof(float)*len,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(data_d) );

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&cost_time,start,stop);
		return flag;
	}
}

#endif