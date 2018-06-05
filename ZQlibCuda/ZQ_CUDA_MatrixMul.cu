#ifndef _ZQ_CUDA_MATRIX_MUL_CU_
#define _ZQ_CUDA_MATRIX_MUL_CU_

#include "ZQ_CUDA_MatrixMul.cuh"

namespace ZQ_CUDA_MatrixMul
{
	__global__ 
	void ZQ_Cuda_MatrixMul_Kernel(const float* A, const float* B, const int wA, const int wB, float* C)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int aBegin = wA*by*blockDim.y;
		int aEnd = aBegin + wA - 1;
		int aStep = BLOCK_SIZE;
		int bBegin = bx*blockDim.x;
		int bStep = BLOCK_SIZE*wB;

		float Csub = 0;
		for(int a = aBegin,b = bBegin; a < aEnd; a+= aStep,b+=bStep)
		{
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
			As[ty][tx] = A[a+wA*ty+tx];
			Bs[ty][tx] = B[b+wB*ty+tx];
			__syncthreads();

	#pragma unroll
			for(int k = 0;k < BLOCK_SIZE;k++)
				Csub += As[ty][k]*Bs[k][tx];

			__syncthreads();
		}
		int c = wB*(BLOCK_SIZE*by+ty)+BLOCK_SIZE*bx+tx;
		C[c] = Csub;
		__syncthreads();
	}

	void cu_MatrixMul_BlockSize(const float* A, const float* B, const int hA, const int wA, const int wB, float* C)
	{
		dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
		dim3 dimGrid(wB/dimBlock.x,hA/dimBlock.y);

		ZQ_Cuda_MatrixMul_Kernel<<<dimGrid,dimBlock>>>(A,B,wA,wB,C);
	}

	extern "C"
	float MatrixMul(const float* A, const float* B, const int hA, const int wA, const int wB, float* C)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		if(hA % BLOCK_SIZE == 0 && wA % BLOCK_SIZE == 0 && wB % BLOCK_SIZE == 0)
		{
			//printf("hA = %d,wA = %d,wB = %d\n",hA,wA,wB);
			//int tmp = 0;
			//scanf("%d",&tmp);
			int sizeA = hA * wA * sizeof(float);
			int sizeB = wA * wB * sizeof(float);
			int sizeC = hA * wB * sizeof(float);
			float* Ad = 0;
			float* Bd = 0;
			float* Cd = 0;
			cudaMalloc((void**)&Ad,sizeA);
			cudaMalloc((void**)&Bd,sizeB);
			cudaMalloc((void**)&Cd,sizeC);

			cudaMemcpy(Ad,A,sizeA,cudaMemcpyHostToDevice);		
			cudaMemcpy(Bd,B,sizeB,cudaMemcpyHostToDevice);

			cu_MatrixMul_BlockSize(Ad,Bd,hA,wA,wB,Cd);

			cudaMemcpy(C,Cd,sizeC,cudaMemcpyDeviceToHost);

			cudaFree(Ad);
			cudaFree(Bd);
			cudaFree(Cd);
		}
		else
		{
			int padding_hA = (hA+BLOCK_SIZE-1) / BLOCK_SIZE * BLOCK_SIZE;
			int padding_wA = (wA+BLOCK_SIZE-1) / BLOCK_SIZE * BLOCK_SIZE;
			int padding_wB = (wB+BLOCK_SIZE-1) / BLOCK_SIZE * BLOCK_SIZE;
		

			float* pA = 0,*pB = 0,*pC = 0;
			bool aflag = false,bflag = false,cflag = false;

			if(padding_hA != hA || padding_wA != wA)
			{
				pA = new float[padding_hA*padding_wA];
				memset(pA,0,sizeof(float)*padding_hA*padding_wA);
				for(int i = 0;i < hA;i++)
					memcpy(pA+i*padding_wA,A+i*wA,sizeof(float)*wA);
				aflag = true;

			}
			else 
				pA = (float*)A;
			if(padding_wA != wA || padding_wB != wB)
			{
				pB = new float[padding_wA*padding_wB];
				memset(pB,0,sizeof(float)*padding_wA*padding_wB);
				for(int i = 0;i < wA;i++)
					memcpy(pB+i*padding_wB,B+i*wB,sizeof(float)*wB);
				bflag = true;

			}
			else
				pB = (float*)B;
			if(padding_hA != hA || padding_wB != wB)
			{
				pC = new float[padding_hA*padding_wB];
				memset(pC,0,sizeof(float)*padding_hA*padding_wB);
				cflag = true;
			}
			else
				pC = C;


			MatrixMul(pA,pB,padding_hA,padding_wA,padding_wB,pC);
			

			if(aflag)
			{
				delete []pA;
				pA = 0;
			}
			if(bflag)
			{
				delete []pB;
				pB = 0;
			}
			if(cflag)
			{
				for(int i = 0;i < hA;i++)
					memcpy(C+i*wB,pC+i*padding_wB,sizeof(float)*wB);
				delete []pC;
				pC = 0;
			}
		}

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
}
#endif