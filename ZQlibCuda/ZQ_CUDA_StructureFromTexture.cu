#ifndef _ZQ_CUDA_STRUCTURE_FROM_TEXTURE_CU_
#define _ZQ_CUDA_STRUCTURE_FROM_TEXTURE_CU_

#include "ZQ_CUDA_StructureFromTexture.cuh"

namespace ZQ_CUDA_StructureFromTexture
{

	void cu_StructureFromTextureImprovedWLS(float* output, const float* input, int width, int height, int nChannels,
		float lambda, int nOuterIter, int nInnerIter, int fsize_for_abs_gradient, float sigma_for_abs_gradient,
		int fsize_for_gradient, float sigma_for_gradient, int fsize_for_contrast, float sigma_for_contrast,
		float norm_for_contrast_num, float norm_for_contrast_denom, float norm_for_data_term, float norm_for_smooth_term, float eps)
	{
		bool need_compute_weightdata = true;
		bool need_compute_psi_phi = true;

		if (norm_for_data_term == 2)
		{
			need_compute_weightdata = false;
		}
		if (norm_for_contrast_denom == 0 && norm_for_contrast_num == 0)
		{
			need_compute_psi_phi = false;
		}

		float* I = 0;
		float* Ix = 0; 
		float* Iy = 0;
		float* absIx = 0;
		float* absIy = 0;
		float* gIx = 0;
		float* gIy = 0;
		float* gAbsIx = 0;
		float* gAbsIy = 0;
		float* psi = 0;
		float* phi = 0;
		float* weightdata = 0;
		float* weightx = 0;
		float* weighty = 0;
		checkCudaErrors(cudaMalloc((void**)&I, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&Ix, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&Iy, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&absIx, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&absIy, sizeof(float)*width*height));
		if (need_compute_psi_phi)
		{
			checkCudaErrors(cudaMalloc((void**)&gIx, sizeof(float)*width*height));
			checkCudaErrors(cudaMalloc((void**)&gIy, sizeof(float)*width*height));
			checkCudaErrors(cudaMalloc((void**)&gAbsIx, sizeof(float)*width*height));
			checkCudaErrors(cudaMalloc((void**)&gAbsIy, sizeof(float)*width*height));
			checkCudaErrors(cudaMalloc((void**)&psi, sizeof(float)*width*height));
			checkCudaErrors(cudaMalloc((void**)&phi, sizeof(float)*width*height));

		}
		if (need_compute_weightdata)
		{
			checkCudaErrors(cudaMalloc((void**)&weightdata, sizeof(float)*width*height));
		}
		checkCudaErrors(cudaMalloc((void**)&weightx, sizeof(float)*width*height));
		checkCudaErrors(cudaMalloc((void**)&weighty, sizeof(float)*width*height));
		/*checkCudaErrors(cudaMemset(I, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(Ix, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(Iy, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(gIx, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(gIy, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(absIx, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(absIy, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(gAbsIx, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(gAbsIy, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(psi, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(phi, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(weightdata, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(weightx, 0, sizeof(float)*width*height));
		checkCudaErrors(cudaMemset(weighty, 0, sizeof(float)*width*height));*/

		

		checkCudaErrors(cudaMemcpy(output, input, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToDevice));

		for (int k = 0; k < nOuterIter; k++)
		{
			for (int c = 0; c < nChannels; c++)
			{
				ZQ_CUDA_ImageProcessing2D::cu_CopyChannel_i(I, output, c, width, height, nChannels);
				
				ZQ_CUDA_ImageProcessing2D::cu_DxForward(Ix, I, width, height, 1);
				ZQ_CUDA_ImageProcessing2D::cu_DyForward(Iy, I, width, height, 1);
				ZQ_CUDA_BaseUtils::cu_Abs(width*height, Ix, absIx);
				ZQ_CUDA_BaseUtils::cu_Abs(width*height, Iy, absIy);

				if (need_compute_psi_phi)
				{
					float* gIx_data = 0, *gIy_data = 0;
					if (fsize_for_gradient <= 0 || sigma_for_abs_gradient <= 0)
					{
						gIx_data = Ix;
						gIy_data = Iy;
					}
					else
					{
						ZQ_CUDA_ImageProcessing2D::cu_GaussianSmoothing2(gIx, Ix, sigma_for_gradient, fsize_for_gradient, width, height, 1);
						ZQ_CUDA_ImageProcessing2D::cu_GaussianSmoothing2(gIy, Iy, sigma_for_gradient, fsize_for_gradient, width, height, 1);
						gIx_data = gIx;
						gIy_data = gIy;
					}

					float* gAbsIx_data = 0, *gAbsIy_data = 0;
					if (fsize_for_abs_gradient <= 0 || sigma_for_abs_gradient <= 0)
					{
						gAbsIx_data = absIx;
						gAbsIy_data = absIy;
					}
					else
					{
						ZQ_CUDA_ImageProcessing2D::cu_GaussianSmoothing2(gAbsIx, absIx, sigma_for_gradient, fsize_for_gradient, width, height, 1);
						ZQ_CUDA_ImageProcessing2D::cu_GaussianSmoothing2(gAbsIy, absIy, sigma_for_gradient, fsize_for_gradient, width, height, 1);
						gAbsIx_data = gAbsIx;
						gAbsIy_data = gAbsIy;
					}

					cu_ComputePsiPhi(psi, phi, gAbsIx_data, gAbsIy_data, gIx_data, gIy_data, width*height, 
						norm_for_contrast_num, norm_for_contrast_denom, eps);
					
					if (!(fsize_for_contrast <= 0 || sigma_for_contrast <= 0))
					{
						ZQ_CUDA_ImageProcessing2D::cu_GaussianSmoothing2(weightx, psi, sigma_for_contrast, fsize_for_contrast, width, height, 1);
						checkCudaErrors(cudaMemcpy(psi, weightx, sizeof(float)*width*height, cudaMemcpyDeviceToDevice));
						ZQ_CUDA_ImageProcessing2D::cu_GaussianSmoothing2(weighty, phi, sigma_for_contrast, fsize_for_contrast, width, height, 1);
						checkCudaErrors(cudaMemcpy(phi, weighty, sizeof(float)*width*height, cudaMemcpyDeviceToDevice));
					}
				}

				if (need_compute_weightdata)
				{
					cu_Compute_weightdata(weightdata, I, input, width*height, nChannels, c,
						norm_for_data_term, eps);
				}

				if (need_compute_psi_phi)
					cu_Compute_weightx_weighty1(weightx, weighty, psi, phi, absIx, absIy, width*height, norm_for_smooth_term, eps);
				else
					cu_Compute_weightx_weighty2(weightx, weighty, absIx, absIy, width*height, norm_for_smooth_term, eps);
				

				//SOR
				double omega = 1.6;
				
				for (int sor_it = 0; sor_it < nInnerIter; sor_it++)
				{
					if (need_compute_weightdata)
					{
						cu_Solve_redblack1(output, input, width, height, nChannels, c, weightdata, weightx, weighty, lambda, omega, true);
						cu_Solve_redblack1(output, input, width, height, nChannels, c, weightdata, weightx, weighty, lambda, omega, false);
					}
					else
					{
						cu_Solve_redblack2(output, input, width, height, nChannels, c, weightx, weighty, lambda, omega, true);
						cu_Solve_redblack2(output, input, width, height, nChannels, c, weightx, weighty, lambda, omega, false);
					}
					
				}
			}
		}

		checkCudaErrors(cudaFree(I));
		checkCudaErrors(cudaFree(Ix));
		checkCudaErrors(cudaFree(Iy));
		checkCudaErrors(cudaFree(absIx));
		checkCudaErrors(cudaFree(absIy));
		if (need_compute_psi_phi)
		{
			checkCudaErrors(cudaFree(gIx));
			checkCudaErrors(cudaFree(gIy));
			checkCudaErrors(cudaFree(gAbsIx));
			checkCudaErrors(cudaFree(gAbsIy));
			checkCudaErrors(cudaFree(psi));
			checkCudaErrors(cudaFree(phi));
		}
		if (need_compute_weightdata)
		{
			checkCudaErrors(cudaFree(weightdata));
		}
		checkCudaErrors(cudaFree(weightx));
		checkCudaErrors(cudaFree(weighty));
	}


	extern "C" float StructureFromTextureImprovedWLS(float* output, const float* input, int width, int height, int nChannels,
		float lambda, int nOuterIter, int nInnerIter, int fsize_for_abs_gradient, float sigma_for_abs_gradient,
		int fsize_for_gradient, float sigma_for_gradient, int fsize_for_contrast, float sigma_for_contrast,
		float norm_for_contrast_num, float norm_for_contrast_denom, float norm_for_data_term, float norm_for_smooth_term, float eps)
	{
		

		float* input_d = 0;
		float* output_d = 0;
		checkCudaErrors(cudaMalloc((void**)&input_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMalloc((void**)&output_d, sizeof(float)*width*height*nChannels));
		checkCudaErrors(cudaMemcpy(input_d, input, sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemset(output_d, 0, sizeof(float)*width*height*nChannels));

		float time = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		cu_StructureFromTextureImprovedWLS(output_d, input_d, width, height, nChannels, lambda, nOuterIter, nInnerIter, 
			fsize_for_abs_gradient, sigma_for_abs_gradient, fsize_for_gradient, sigma_for_gradient, fsize_for_contrast, sigma_for_contrast, 
			norm_for_contrast_num, norm_for_contrast_denom, norm_for_data_term, norm_for_smooth_term, eps);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);


		checkCudaErrors(cudaMemcpy(output, output_d, sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(input_d));
		checkCudaErrors(cudaFree(output_d));

		
		return time;
	}

	__global__ void Compute_psi_phi_Kernel(float* psi, float* phi, const float* gAbsIx, const float* gAbsIy,
		const float* gIx, const float* gIy, int nPixels,
		float norm_for_contrast_num, float norm_for_contrast_denom, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;

		
		float psi_num = 0, psi_denom = 0;
		float phi_num = 0, phi_denom = 0;
		if (norm_for_contrast_num == 0)
		{
			psi_num = 1;
			phi_num = 1;
		}
		else if (norm_for_contrast_num == 1)
		{
			psi_num = gAbsIx[x];
			phi_num = gAbsIy[x];
		}
		else if (norm_for_contrast_num == 2)
		{
			psi_num = gAbsIx[x] * gAbsIx[x];
			phi_num = gAbsIy[x] * gAbsIy[x];
		}
		else
		{
			psi_num = pow(gAbsIx[x], norm_for_contrast_num);
			phi_num = pow(gAbsIy[x], norm_for_contrast_num);
		}

		if (norm_for_contrast_denom == 0)
		{
			psi_denom = 1;
			phi_denom = 1;
		}
		else if (norm_for_contrast_denom == 1)
		{
			psi_denom = fabs(gIx[x]) + eps;
			phi_denom = fabs(gIy[x]) + eps;
		}
		else if (norm_for_contrast_denom == 2)
		{
			psi_denom = gIx[x] * gIx[x] + eps;
			phi_denom = gIy[x] * gIy[x] + eps;
		}
		else
		{
			psi_denom = pow(fabs(gIx[x]), norm_for_contrast_denom) + eps;
			phi_denom = pow(fabs(gIy[x]), norm_for_contrast_denom) + eps;
		}
		psi[x] = psi_num / psi_denom;
		phi[x] = phi_num / phi_denom;
		
	}

	void cu_ComputePsiPhi(float* psi, float* phi, const float* gAbsIx, const float* gAbsIy, 
		const float* gIx, const float* gIy, int nPixels,
		float norm_for_contrast_num, float norm_for_contrast_denom, float eps)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((nPixels - 1) / blockSize.x + 1, 1);

		Compute_psi_phi_Kernel << <gridSize, blockSize >> >(psi,phi, gAbsIx, gAbsIy, gIx, gIy, nPixels, norm_for_contrast_num, norm_for_contrast_denom, eps);
	}

	__global__ void Compute_weightdata_Kernel(float* weightdata, const float* I, const float* input, int nPixels, int nChannels, int c,
		float norm_for_data_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		
		if (norm_for_data_term == 2)
		{
			weightdata[x] = 1;
		}
		else if (norm_for_data_term == 1)
		{
			weightdata[x] = 1.0f / (fabs(I[x] - input[x*nChannels + c]) + eps);
		}
		else
		{
			weightdata[x] = pow(fabs(I[x] - input[x*nChannels + c]) + eps, norm_for_data_term - 2);
		}
	}

	void cu_Compute_weightdata(float* weightdata, const float* I, const float* input, int nPixels, int nChannels, int c,
		float norm_for_data_term, float eps)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((nPixels - 1) / blockSize.x + 1, 1);

		Compute_weightdata_Kernel << <gridSize, blockSize >> >(weightdata, I, input, nPixels, nChannels, c, norm_for_data_term, eps);
	}
	

	__global__ void Compute_weightx_weighty1_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;
		
		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		if (norm_for_smooth_term == 2)
		{
			weightx[x] = psi[x];
			weighty[x] = phi[x];
		}
		else if (norm_for_smooth_term == 1)
		{
			weightx[x] = psi[x] / (absIx[x] + eps);
			weighty[x] = phi[x] / (absIy[x] + eps);
		}
		else if (norm_for_smooth_term == 0)
		{
			weightx[x] = psi[x] / (absIx[x] * absIx[x] + eps);
			weighty[x] = phi[x] / (absIy[x] * absIy[x] + eps);
		}
		else
		{
			weightx[x] = psi[x] / (pow(absIx[x], 2.0f - norm_for_smooth_term) + eps);
			weighty[x] = phi[x] / (pow(absIy[x], 2.0f - norm_for_smooth_term) + eps);
		}
	}

	__global__ void Compute_weightx_weighty1_norm2_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		weightx[x] = psi[x];
		weighty[x] = phi[x];
	}

	__global__ void Compute_weightx_weighty1_norm1_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		
		weightx[x] = psi[x] / (absIx[x] + eps);
		weighty[x] = phi[x] / (absIy[x] + eps);
	}

	__global__ void Compute_weightx_weighty1_norm0_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		
		weightx[x] = psi[x] / (absIx[x] * absIx[x] + eps);
		weighty[x] = phi[x] / (absIy[x] * absIy[x] + eps);
	}

	__global__ void Compute_weightx_weighty1_normother_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		
		weightx[x] = psi[x] / (pow(absIx[x], 2.0f - norm_for_smooth_term) + eps);
		weighty[x] = phi[x] / (pow(absIy[x], 2.0f - norm_for_smooth_term) + eps);
	}

	__global__ void Compute_weightx_weighty2_Kernel(float* weightx, float* weighty, 
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		if (norm_for_smooth_term == 2)
		{
			weightx[x] = 1.0f;
			weighty[x] = 1.0f;
		}
		else if (norm_for_smooth_term == 1)
		{
			weightx[x] = 1.0f / (absIx[x] + eps);
			weighty[x] = 1.0f / (absIy[x] + eps);
		}
		else if (norm_for_smooth_term == 0)
		{
			weightx[x] = 1.0f / (absIx[x] * absIx[x] + eps);
			weighty[x] = 1.0f / (absIy[x] * absIy[x] + eps);
		}
		else
		{
			weightx[x] = 1.0f / (pow(absIx[x], 2.0f - norm_for_smooth_term) + eps);
			weighty[x] = 1.0f / (pow(absIy[x], 2.0f - norm_for_smooth_term) + eps);
		}
	}

	__global__ void Compute_weightx_weighty2_norm2_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;

		weightx[x] = 1.0f;
		weighty[x] = 1.0f;
	}

	__global__ void Compute_weightx_weighty2_norm1_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		weightx[x] = 1.0f / (absIx[x] + eps);
		weighty[x] = 1.0f / (absIy[x] + eps);
	}

	__global__ void Compute_weightx_weighty2_norm0_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		
		weightx[x] = 1.0f / (absIx[x] * absIx[x] + eps);
		weighty[x] = 1.0f / (absIy[x] * absIy[x] + eps);
	}

	__global__ void Compute_weightx_weighty2_normother_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;

		int x = bx*blockDim.x + tx;
		if (x >= nPixels)
			return;
		
		weightx[x] = 1.0f / (pow(absIx[x], 2.0f - norm_for_smooth_term) + eps);
		weighty[x] = 1.0f / (pow(absIy[x], 2.0f - norm_for_smooth_term) + eps);
	}

	void cu_Compute_weightx_weighty1(float* weightx, float* weighty, const float* psi, const float* phi, 
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((nPixels - 1) / blockSize.x + 1, 1);

		if (norm_for_smooth_term == 2)
			Compute_weightx_weighty1_norm2_Kernel << <gridSize, blockSize >> >(weightx, weighty, psi, phi, absIx, absIy, nPixels, norm_for_smooth_term, eps);
		else if (norm_for_smooth_term == 1)
			Compute_weightx_weighty1_norm1_Kernel << <gridSize, blockSize >> >(weightx, weighty, psi, phi, absIx, absIy, nPixels, norm_for_smooth_term, eps);
		else if (norm_for_smooth_term == 0)
			Compute_weightx_weighty1_norm0_Kernel << <gridSize, blockSize >> >(weightx, weighty, psi, phi, absIx, absIy, nPixels, norm_for_smooth_term, eps);
		else
			Compute_weightx_weighty1_normother_Kernel << <gridSize, blockSize >> >(weightx, weighty, psi, phi, absIx, absIy, nPixels, norm_for_smooth_term, eps);
	}

	void cu_Compute_weightx_weighty2(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
	{
		dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE, 1);
		dim3 gridSize((nPixels - 1) / blockSize.x + 1, 1);

		if (norm_for_smooth_term == 2)
			Compute_weightx_weighty2_norm2_Kernel << <gridSize, blockSize >> >(weightx, weighty, absIx, absIy, nPixels, norm_for_smooth_term, eps);
		else if (norm_for_smooth_term == 1)
			Compute_weightx_weighty2_norm1_Kernel << <gridSize, blockSize >> >(weightx, weighty, absIx, absIy, nPixels, norm_for_smooth_term, eps);
		else if (norm_for_smooth_term == 0)
			Compute_weightx_weighty2_norm0_Kernel << <gridSize, blockSize >> >(weightx, weighty, absIx, absIy, nPixels, norm_for_smooth_term, eps);
		else
			Compute_weightx_weighty2_normother_Kernel << <gridSize, blockSize >> >(weightx, weighty, absIx, absIy, nPixels, norm_for_smooth_term, eps);
	}

	__global__ void Solve_redblack1_Kernel(float* output, const float* input, int width, int height, int nChannels, int c, 
		const float* weightdata, const float* weightx, const float* weighty, 
		float lambda, float omega, bool redflag)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;
		if (x >= width || y >= height)
			return;

		if ((y + x) % 2 == redflag)
			return;
		
		int offset = y*width + x;
		int slice = width*nChannels;
		int offset_c = offset*nChannels + c;
		float coeff = 0, sigma = 0, weight = 0;
		if (y > 0)
		{
			weight = lambda*weighty[offset - width];
			coeff += weight;
			sigma += weight * output[offset_c - slice];
		}
		if (y < height - 1)
		{
			weight = lambda*weighty[offset];
			coeff += weight;
			sigma += weight*output[offset_c + slice];
		}
		if (x > 0)
		{
			weight = lambda*weightx[offset - 1];
			coeff += weight;
			sigma += weight*output[offset_c - nChannels];
		}
		if (x < width - 1)
		{
			weight = lambda*weightx[offset];
			coeff += weight;
			sigma += weight*output[offset_c + nChannels];
		}
		coeff += weightdata[offset];
		sigma += weightdata[offset] * input[offset_c];
		if (coeff > 0)
			output[offset_c] = sigma / coeff*omega + output[offset_c] * (1 - omega);
	}

	__global__ void Solve_redblack1_new_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty,
		float lambda, float omega, bool redflag)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;

		int start = redflag ? y % 2 : y % 2 + 1;
		x = x * 2 + start;
		
		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		int slice = width*nChannels;
		int offset_c = offset*nChannels + c;
		float coeff = 0, sigma = 0, weight = 0;
		if (y > 0)
		{
			weight = lambda*weighty[offset - width];
			coeff += weight;
			sigma += weight * output[offset_c - slice];
		}
		if (y < height - 1)
		{
			weight = lambda*weighty[offset];
			coeff += weight;
			sigma += weight*output[offset_c + slice];
		}
		if (x > 0)
		{
			weight = lambda*weightx[offset - 1];
			coeff += weight;
			sigma += weight*output[offset_c - nChannels];
		}
		if (x < width - 1)
		{
			weight = lambda*weightx[offset];
			coeff += weight;
			sigma += weight*output[offset_c + nChannels];
		}
		coeff += weightdata[offset];
		sigma += weightdata[offset] * input[offset_c];
		if (coeff > 0)
			output[offset_c] = sigma / coeff*omega + output[offset_c] * (1 - omega);
	}

	__global__ void Solve_redblack2_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty,
		float lambda, float omega, bool redflag)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;
		if (x >= width || y >= height)
			return;

		if ((y + x) % 2 == redflag)
			return;

		int offset = y*width + x;
		int slice = width*nChannels;
		int offset_c = offset*nChannels + c;
		float coeff = 0, sigma = 0, weight = 0;
		if (y > 0)
		{
			weight = lambda*weighty[offset - width];
			coeff += weight;
			sigma += weight * output[offset_c - slice];
		}
		if (y < height - 1)
		{
			weight = lambda*weighty[offset];
			coeff += weight;
			sigma += weight*output[offset_c + slice];
		}
		if (x > 0)
		{
			weight = lambda*weightx[offset - 1];
			coeff += weight;
			sigma += weight*output[offset_c - nChannels];
		}
		if (x < width - 1)
		{
			weight = lambda*weightx[offset];
			coeff += weight;
			sigma += weight*output[offset_c + nChannels];
		}
		coeff += 1;
		sigma += input[offset_c];
		if (coeff > 0)
			output[offset_c] = sigma / coeff*omega + output[offset_c] * (1 - omega);
	}

	__global__ void Solve_redblack2_new_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty,
		float lambda, float omega, bool redflag)
	{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int x = bx*blockDim.x + tx;
		int y = by*blockDim.y + ty;

		int start = redflag ? y % 2 : y % 2 + 1;
		x = x * 2 + start;


		if (x >= width || y >= height)
			return;

		int offset = y*width + x;
		int slice = width*nChannels;
		int offset_c = offset*nChannels + c;
		float coeff = 0, sigma = 0, weight = 0;
		if (y > 0)
		{
			weight = lambda*weighty[offset - width];
			coeff += weight;
			sigma += weight * output[offset_c - slice];
		}
		if (y < height - 1)
		{
			weight = lambda*weighty[offset];
			coeff += weight;
			sigma += weight*output[offset_c + slice];
		}
		if (x > 0)
		{
			weight = lambda*weightx[offset - 1];
			coeff += weight;
			sigma += weight*output[offset_c - nChannels];
		}
		if (x < width - 1)
		{
			weight = lambda*weightx[offset];
			coeff += weight;
			sigma += weight*output[offset_c + nChannels];
		}
		coeff += 1;
		sigma += input[offset_c];
		if (coeff > 0)
			output[offset_c] = sigma / coeff*omega + output[offset_c] * (1 - omega);
	}

	void cu_Solve_redblack1(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool redflag)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Solve_redblack1_Kernel << <gridSize, blockSize >> >(output, input, width, height, nChannels, c, weightdata, weightx, weighty, lambda, omega, redflag);
	}

	void cu_Solve_redblack1_new(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool redflag)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize(((width+1)/2 - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Solve_redblack1_new_Kernel << <gridSize, blockSize >> >(output, input, width, height, nChannels, c, weightdata, weightx, weighty, lambda, omega, redflag);
	}

	void cu_Solve_redblack2(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty, float lambda, float omega, bool redflag)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Solve_redblack2_Kernel << <gridSize, blockSize >> >(output, input, width, height, nChannels,c, weightx, weighty, lambda, omega, redflag);
	}

	void cu_Solve_redblack2_new(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty, float lambda, float omega, bool redflag)
	{
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridSize(((width+1)/2 - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		Solve_redblack2_new_Kernel << <gridSize, blockSize >> >(output, input, width, height, nChannels, c, weightx, weighty, lambda, omega, redflag);
	}
}

#endif