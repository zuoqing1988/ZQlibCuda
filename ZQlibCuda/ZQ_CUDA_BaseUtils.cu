#ifndef _ZQ_CUDA_BASE_UTILS_CU_
#define _ZQ_CUDA_BASE_UTILS_CU_

#include "ZQ_CUDA_BaseUtils.cuh"

namespace ZQ_CUDA_BaseUtils
{


	/*********************/
	__global__
	void Find_MAX_Value_kernel(const int num, const float* vals, float* result)
	{
		int idx = threadIdx.x;

		__shared__ float tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		if(idx < num)
			tmp_val[idx] = vals[idx];

		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx + skip; i < num;i += skip)
		{
			if(tmp_val[idx] < vals[i])
				tmp_val[idx] = vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				if(result[0] < tmp_val[i])
					result[0] = tmp_val[i];
			}
		}
	}
	
	__global__
	void Find_MAX_Value_kernel(const int num, const int* vals, int* result)
	{
		int idx = threadIdx.x;

		__shared__ int tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		if(idx < num)
			tmp_val[idx] = vals[idx];

		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx + skip; i < num;i += skip)
		{
			if(tmp_val[idx] < vals[i])
				tmp_val[idx] = vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				if(result[0] < tmp_val[i])
					result[0] = tmp_val[i];
			}
		}
	}

	__global__
	void Find_MIN_Value_kernel(const int num, const float* vals, float* result)
	{
		int idx = threadIdx.x;

		__shared__ float tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		if(idx < num)
			tmp_val[idx] = vals[idx];

		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx + skip; i < num;i += skip)
		{
			if(tmp_val[idx] > vals[i])
				tmp_val[idx] = vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				if(result[0] > tmp_val[i])
					result[0] = tmp_val[i];
			}
		}
	}
	
	__global__
	void Find_MIN_Value_kernel(const int num, const int* vals, int* result)
	{
		int idx = threadIdx.x;

		__shared__ int tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		if(idx < num)
			tmp_val[idx] = vals[idx];

		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx + skip; i < num;i += skip)
		{
			if(tmp_val[idx] > vals[i])
				tmp_val[idx] = vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				if(result[0] > tmp_val[i])
					result[0] = tmp_val[i];
			}
		}
	}
	
	__global__
	void SUM_kernel(const int num, const int* vals, int* result)
	{
		int idx = threadIdx.x;

		__shared__ int tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		tmp_val[idx] = 0;
	
		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx; i < num;i += skip)
		{
			tmp_val[idx] += vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				result[0] += tmp_val[i];
			}
		}
	}
	
	__global__
	void SUM_kernel(const int num, const float* vals, float* result)
	{
		int idx = threadIdx.x;

		__shared__ float tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		tmp_val[idx] = 0;
	
		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx; i < num;i += skip)
		{
			tmp_val[idx] += vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				result[0] += tmp_val[i];
			}
		}
	}
	
	__global__
	void SUM_Square_kernel(const int num, const int* vals, int* result)
	{
		int idx = threadIdx.x;

		__shared__ int tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		tmp_val[idx] = 0;
	
		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx; i < num;i += skip)
		{
			tmp_val[idx] += vals[i]*vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				result[0] += tmp_val[i];
			}
		}
	}
	
	__global__
	void SUM_Square_kernel(const int num, const float* vals, float* result)
	{
		int idx = threadIdx.x;

		__shared__ float tmp_val[BLOCK_SIZE*BLOCK_SIZE];

		tmp_val[idx] = 0;
	
		int skip = BLOCK_SIZE*BLOCK_SIZE;

		for(int i = idx; i < num;i += skip)
		{
			tmp_val[idx] += vals[i]*vals[i];
		}

		__syncthreads();

		if(idx == 0)
		{
			result[0] = tmp_val[0];
			for(int i = 1;i < num && i < skip;i++)
			{
				result[0] += tmp_val[i];
			}
		}
	}
	
	__global__
	void Square_kernel(const int num, const int* input, int* output)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx >= num)
			return ;
			
		output[idx] = input[idx]*input[idx];
	}
	
	__global__
	void Square_kernel(const int num, const float* input, float* output)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx >= num)
			return ;
			
		output[idx] = input[idx]*input[idx];
	}
	
	__global__
	void Abs_kernel(const int num, const int* input, int* output)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx >= num)
			return ;
			
		output[idx] = input[idx] >= 0 ? input[idx] : -input[idx];
	}
	
	__global__
	void Abs_kernel(const int num, const float* input, float* output)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		if(idx >= num)
			return ;
			
		output[idx] = input[idx] >= 0 ? input[idx] : -input[idx];
	}

	/////////////////////////////////////


	bool cu_Find_MAX_Value(const int num, const float* vals, float& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		float* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(float)*1));	
		Find_MAX_Value_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(float)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	bool cu_Find_MAX_Value(const int num, const int* vals, int& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		int* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(int)*1));	
		Find_MAX_Value_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(int)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}

	bool cu_Find_MIN_Value(const int num, const float* vals, float& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		float* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(float)*1));
		Find_MIN_Value_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(float)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	bool cu_Find_MIN_Value(const int num, const int* vals, int& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		int* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(int)*1));
		Find_MIN_Value_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(int)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	bool cu_SUM(const int num, const int* vals, int& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		int* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(int)*1));
		SUM_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(int)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	bool cu_SUM(const int num, const float* vals, float& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		float* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(float)*1));
		SUM_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(float)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	bool cu_SUM_Square(const int num, const int* vals, int& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		int* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(int)*1));
		SUM_Square_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(int)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	bool cu_SUM_Square(const int num, const float* vals, float& result)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		if(num <=0 )
			return false;

		float* result_d = 0;
		checkCudaErrors( cudaMalloc((void**)&result_d,sizeof(float)*1));
		SUM_Square_kernel<<<gridSize,blockSize>>>(num,vals,result_d);
		checkCudaErrors( cudaMemcpy(&result,result_d,sizeof(float)*1,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(result_d) );
		return true;
	}
	
	void cu_Square(const int num, const int* input, int* output)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);

		Square_kernel<<<gridSize,blockSize>>>(num,input,output);
	}
	
	void cu_Square(const int num, const float* input, float* output)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);

		Square_kernel<<<gridSize,blockSize>>>(num,input,output);
	}
	
	void cu_Abs(const int num, const int* input, int* output)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);

		Abs_kernel<<<gridSize,blockSize>>>(num,input,output);
	}
	
	void cu_Abs(const int num, const float* input, float* output)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);

		Abs_kernel<<<gridSize,blockSize>>>(num,input,output);
	}

	

	extern "C"
	bool Find_MAX_Value(const int num, const float* vals, float& result)
	{
		if(num <= 0)
			return false;

		float* vals_d = 0;
		checkCudaErrors( cudaMalloc((void**)&vals_d,sizeof(float)*num) );
		checkCudaErrors( cudaMemcpy(vals_d,vals,sizeof(float)*num,cudaMemcpyHostToDevice) );
		cu_Find_MAX_Value(num,vals_d,result);
		checkCudaErrors( cudaFree(vals_d) );
		return true;
	}


	extern "C"
	bool Find_MIN_Value(const int num, const float* vals, float& result)
	{
		if(num <= 0)
			return false;

		float* vals_d = 0;
		checkCudaErrors( cudaMalloc((void**)&vals_d,sizeof(float)*num) );
		checkCudaErrors( cudaMemcpy(vals_d,vals,sizeof(float)*num,cudaMemcpyHostToDevice) );
		cu_Find_MIN_Value(num,vals_d,result);
		checkCudaErrors( cudaFree(vals_d) );
		return true;
	}
	
	/***************************************************************************************************************/
	
	__global__
	void compute_bucket_stored_num_step1_kernel(const int num, const int* coord_in_which_bucket, bool* visited, int* head, bool* finished)
	{
		
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		
		if(idx >= num)
			return;
		
		
		if(!visited[idx])
		{
			int bucket_idx = coord_in_which_bucket[idx];
			head[bucket_idx] = idx;
			finished[0] = false;
		}
	}
	
	__global__
	void compute_bucket_stored_num_step2_kernel(const int num, const int* coord_in_which_bucket, int* bucket_stored_num, bool* visited, int* head)
	{
		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		
		if(idx >= num)
			return;
		
		if(!visited[idx])
		{
			int bucket_idx = coord_in_which_bucket[idx];
			if(head[bucket_idx] == idx)
			{
				visited[idx] = true;
				bucket_stored_num[bucket_idx]++;
			}
		}
	}
	
	__global__
	void compute_bucket_stored_offset_kernel(const int bucket_num, const int b_slice, const int* bucket_stored_num, int* bucket_stored_offset)
	{
		int idx = threadIdx.x;
		int STEP_SIZE = blockDim.x;
		
		__shared__ int s_sum[BLOCK_SIZE*BLOCK_SIZE];
		__shared__ int s_off[BLOCK_SIZE*BLOCK_SIZE];
		s_sum[idx] = 0;
		
		for(int i = b_slice*idx;i < bucket_num && i < b_slice*(idx+1);i++)
		{
			s_sum[idx] += bucket_stored_num[i];
		}
		
		__syncthreads();
		
		if(idx == 0)
		{	
			int total_sum = 0;
			s_off[0] = 0;
			for(int i = 1;i < STEP_SIZE;i++)
			{
				total_sum += s_sum[i-1];
				s_off[i] = total_sum;
			}	
		}
		
		__syncthreads();
		
		bucket_stored_offset[b_slice*idx] = s_off[idx];
		
		for(int i = b_slice*idx+1;i < bucket_num && i < b_slice*(idx+1);i++)
		{
			bucket_stored_offset[i] = bucket_stored_offset[i-1]+bucket_stored_num[i-1];
		}
	}
	
	__global__
	void compute_bucket_stored_index_step1_kernel(const int num, const int* coord_in_which_bucket, bool* visited, int* head, bool* finished)
	{
		int idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if(idx >= num)
			return;
		
		if(!visited[idx])
		{
			int bucket_idx = coord_in_which_bucket[idx];
			head[bucket_idx] = idx;
			finished[0] = false;
		}
	}
	
	__global__
	void compute_bucket_stored_index_step2_kernel(const int num, const int* coord_in_which_bucket, int* bucket_stored_num, const int* bucket_stored_offset, 
			int* bucket_stored_index,  bool* visited, int* head)
	{
		int idx = (blockIdx.x*blockDim.x+threadIdx.x);
		
		if(idx >= num)
			return;
		
		if(!visited[idx])
		{
			int bucket_idx = coord_in_which_bucket[idx];
			if(head[bucket_idx] == idx)
			{
				visited[idx] = true;
				int offset = bucket_stored_offset[bucket_idx];
				int count = bucket_stored_num[bucket_idx];
				
				bucket_stored_index[offset+count] = idx;
				count ++;
				bucket_stored_num[bucket_idx] = count;
			}
		}
	}
	
	void cu_Compute_bucket_stored_num(const int num, const int bucket_num,const int* coord_in_which_bucket, int* bucket_stored_num)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		
		bool finished = true;
		bool* visited = 0;
		int* head = 0;
		bool* finished_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&visited,sizeof(bool)*num) );
		checkCudaErrors( cudaMalloc((void**)&head,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMalloc((void**)&finished_d,sizeof(bool)*1) );
		checkCudaErrors( cudaMemset(visited,0,sizeof(bool)*num) );
		checkCudaErrors( cudaMemset(head,0,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMemset(finished_d,0,sizeof(bool)*1) );
		
		while(true)
		{
			finished = true;
			checkCudaErrors( cudaMemset(finished_d,1,sizeof(bool)*1) );
			
			compute_bucket_stored_num_step1_kernel<<<gridSize,blockSize>>>(num,coord_in_which_bucket,visited,head,finished_d);
			
			checkCudaErrors( cudaMemcpy(&finished,finished_d,sizeof(bool)*1,cudaMemcpyDeviceToHost) );
			
			if(finished)
				break;
			
			compute_bucket_stored_num_step2_kernel<<<gridSize,blockSize>>>(num,coord_in_which_bucket,bucket_stored_num,visited,head);
		}
		
		checkCudaErrors( cudaFree(finished_d) );
		checkCudaErrors( cudaFree(visited) );
		checkCudaErrors( cudaFree(head) );
		
	}
	
	void cu_Compute_bucket_stored_offset(const int bucket_num, const int* bucket_stored_num, int* bucket_stored_offset)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
		dim3 gridSize(1,1,1);
		
		int b_slice = (bucket_num+blockSize.x-1)/blockSize.x;
		compute_bucket_stored_offset_kernel<<<gridSize,blockSize>>>(bucket_num,b_slice,bucket_stored_num,bucket_stored_offset);
	}
	
	void cu_Compute_bucket_stored_index(const int num, const int bucket_num, const int* coord_in_which_bucket, int* bucket_stored_num, const int* bucket_stored_offset, 
			int* bucket_stored_index)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE,1);
		dim3 gridSize((num+blockSize.x-1)/blockSize.x,1);
		
		bool finished = true;
		bool* visited = 0;
		int* head = 0;
		bool* finished_d = 0;
		
		checkCudaErrors( cudaMalloc((void**)&visited,sizeof(bool)*num) );
		checkCudaErrors( cudaMalloc((void**)&head,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMalloc((void**)&finished_d,sizeof(bool)*1) );
		checkCudaErrors( cudaMemset(visited,0,sizeof(bool)*num) );
		checkCudaErrors( cudaMemset(head,0,sizeof(int)*bucket_num) );
		checkCudaErrors( cudaMemset(finished_d,0,sizeof(bool)*1) );
		
		checkCudaErrors( cudaMemset(bucket_stored_num,0,sizeof(int)*bucket_num) );
		
		while(true)
		{
			finished = true;
			checkCudaErrors( cudaMemcpy(finished_d,&finished,sizeof(bool)*1,cudaMemcpyHostToDevice) );
			
			compute_bucket_stored_index_step1_kernel<<<gridSize,blockSize>>>(num,coord_in_which_bucket,visited,head,finished_d);
			
			checkCudaErrors( cudaMemcpy(&finished,finished_d,sizeof(bool)*1,cudaMemcpyDeviceToHost) );
			
			if(finished)
				break;
			
			compute_bucket_stored_index_step2_kernel<<<gridSize,blockSize>>>(num,coord_in_which_bucket,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
						visited,head);
		}
		
		checkCudaErrors( cudaFree(finished_d) );
		checkCudaErrors( cudaFree(visited) );
		checkCudaErrors( cudaFree(head) );
		
	}
	
	void cu_Distribute_Bucket(const int num, const int bucket_num, int* bucket_stored_num, int* bucket_stored_offset, int* bucket_stored_index, const int* coord_in_which_bucket)
	{
		cu_Compute_bucket_stored_num(num,bucket_num,coord_in_which_bucket,bucket_stored_num);
		
		cu_Compute_bucket_stored_offset(bucket_num,bucket_stored_num,bucket_stored_offset);
		
		cu_Compute_bucket_stored_index(num,bucket_num,coord_in_which_bucket,bucket_stored_num,bucket_stored_offset,bucket_stored_index);
	}
}


#endif