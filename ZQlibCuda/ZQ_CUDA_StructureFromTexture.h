#ifndef _ZQ_CUDA_STRUCTURE_FROM_TEXTURE_H_
#define _ZQ_CUDA_STRUCTURE_FROM_TEXTURE_H_
#pragma once

namespace ZQ_CUDA_StructureFromTexture
{
	extern "C" float StructureFromTextureImprovedWLS(float* output, const float* input, int width, int height, int nChannels,
		float lambda, int nOuterIter, int nInnerIter, int fsize_for_abs_gradient, float sigma_for_abs_gradient,
		int fsize_for_gradient, float sigma_for_gradient, int fsize_for_contrast, float sigma_for_contrast,
		float norm_for_contrast_num, float norm_for_contrast_denom, float norm_for_data_term, float norm_for_smooth_term, float eps);
}

#endif