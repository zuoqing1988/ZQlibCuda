#ifndef _ZQ_CUDA_ADVECTION_2D_H_
#define _ZQ_CUDA_ADVECTION_2D_H_
#pragma once

namespace ZQ_CUDA_Advection2D
{
	extern "C" void ZQ_Cuda_Prepare_Advection2D(const unsigned int width, const unsigned int height, const float voxelSize, const unsigned int steps, const float deltatt);

	/* velocity		: [width*height*2], the velocity field 
	*  occupy		: [width*height], the occupy field
	*  inVelocity	: [width*height*2], the velocity field to be advected
	*  outVelocity	: [width*height*2], the output 
	*/
	extern "C" void ZQ_Cuda_Velocity_Advection2D_inRegular_outRegular(const float* velocity, const bool* occupy, const float* inVelocity, float* outVelocity);

	/* velocity		: [width*height*2], the velocity field 
	*  occupy		: [width*height], the occupy field
	*  inVelocity	: [width*height*2], the velocity field to be advected
	*  out_mac_u	: [(width+1)*height], the output u (MAC grid)
	*  out_mac_v	: [width*(height+1)], the output v (MAC grid)
	*/
	extern "C" void ZQ_Cuda_Velocity_Advection2D_inRegular_outMAC(const float* velocity, const bool* occupy, const float* inVelocity, float* out_mac_u, float* out_mac_v);

	/* vel_mac_u	: [(width+1)*height], the velocity field u (MAC grid)
	*  vel_mac_v	: [width*(height+1)], the velocity field v (MAC grid)
	*  occupy		: [width*height], the occupy field
	*  in_mac_u		: [(width+1)*height], the velocity field u to be advected (MAC grid)
	*  in_mac_v		: [width*(height+1)], the velocity field v to be advected (MAC grid)
	*  out_mac_u	: [(width+1)*height], the output u (MAC grid)
	*  out_mac_v	: [width*(height+1)], the output v (MAC grid)
	*/
	extern "C" void ZQ_Cuda_Velocity_Advection2D_inMAC_outMAC(const float* vel_mac_u, const float* vel_mac_v, const bool* occupy, const float* in_mac_u, const float* in_mac_v, 
		float* out_mac_u, float* out_mac_v);

	/* velocity		: [width*height*2], the velocity field 
	*  occupy		: [width*height], the occupy field
	*  input		: [width*height*2], the scalar field to be advected
	*  output		: [width*height*2], output
	*/
	extern "C" void ZQ_Cuda_Scalar_Advection2D_Regular_Velocity(const float* velocity, const bool* occupy, const float* input, float* output);

	/*  vel_mac_u	: [(width+1)*height], the velocity field u (MAC grid)
	*  vel_mac_v	: [width*(height+1)], the velocity field v (MAC grid)
	*  occupy		: [width*height], the occupy field
	*  input		: [width*height*2], the scalar field to be advected
	*  output		: [width*height*2], output
	*/
	extern "C" void ZQ_Cuda_Scalar_Advection2D_MAC_Velocity(const float* in_mac_u, const float* in_mac_v, const bool* occupy, const float* input, float* output);

}

#endif