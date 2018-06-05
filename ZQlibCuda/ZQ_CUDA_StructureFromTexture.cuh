#ifndef _ZQ_CUDA_STRUCTURE_FROM_TEXTURE_CUH_
#define _ZQ_CUDA_STRUCTURE_FROM_TEXTURE_CUH_

#include "ZQlibCudaDefines.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"
#include "ZQ_CUDA_BaseUtils.cuh"
#include <stdio.h>
#include <stdlib.h>

namespace ZQ_CUDA_StructureFromTexture
{
	void cu_StructureFromTextureImprovedWLS(float* output, const float* input, int width, int height, int nChannels,
		float lambda, int nOuterIter, int nInnerIter, int fsize_for_abs_gradient, float sigma_for_abs_gradient,
		int fsize_for_gradient, float sigma_for_gradient, int fsize_for_contrast, float sigma_for_contrast,
		float norm_for_contrast_num, float norm_for_contrast_denom, float norm_for_data_term, float norm_for_smooth_term, float eps);

	/******************/

	__global__ void Compute_psi_phi_Kernel(float* psi, float* phi, const float* gAbsIx, const float* gAbsIy,
		const float* gIx, const float* gIy, int nPixels,
		float norm_for_contrast_num, float norm_for_contrast_denom, float eps);

	void cu_ComputePsiPhi(float* psi, float* phi, const float* gAbsIx, const float* gAbsIy,
		const float* gIx, const float* gIy, int nPixels,
		float norm_for_contrast_num, float norm_for_contrast_denom, float eps);

	__global__ void Compute_weightdata_Kernel(float* weightdata, const float* I, const float* input, int nPixels, int nChannels, int c,
		float norm_for_data_term, float eps);

	void cu_Compute_weightdata(float* weightdata, const float* I, const float* input, int nPixels, int nChannels, int c,
		float norm_for_data_term, float eps);

	__global__ void Compute_weightx_weighty1_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty1_norm2_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty1_norm1_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty1_norm0_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty1_normother_Kernel(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty2_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty2_norm2_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty2_norm1_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty2_norm0_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Compute_weightx_weighty2_normother_Kernel(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	void cu_Compute_weightx_weighty1(float* weightx, float* weighty, const float* psi, const float* phi,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	void cu_Compute_weightx_weighty2(float* weightx, float* weighty,
		const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps);

	__global__ void Solve_redblack1_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool reflag);

	__global__ void Solve_redblack1_new_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool reflag);

	__global__ void Solve_redblack2_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty, float lambda, float omega, bool reflag);

	__global__ void Solve_redblack2_new_Kernel(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty, float lambda, float omega, bool reflag);

	void cu_Solve_redblack1(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool redflag);

	void cu_Solve_redblack1_new(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool redflag);

	void cu_Solve_redblack2(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty, float lambda, float omega, bool redflag);

	void cu_Solve_redblack2_new(float* output, const float* input, int width, int height, int nChannels, int c,
		const float* weightx, const float* weighty, float lambda, float omega, bool redflag);
}

#endif