#ifndef _ZQ_CUDA_ADVECTION_2D_CU_
#define _ZQ_CUDA_ADVECTION_2D_CU_

#include "ZQ_CUDA_Advection2D.cuh"


	
	
	
namespace ZQ_CUDA_Advection2D
{
	texture<float2,2,cudaReadModeElementType> tex_velocity_regular;
	texture<float,2,cudaReadModeElementType> tex_velocity_MAC_u;
	texture<float,2,cudaReadModeElementType> tex_velocity_MAC_v;
	texture<unsigned char,2,cudaReadModeElementType> tex_occupy;
	texture<float,2,cudaReadModeElementType> tex_inputVelocity_mac_u;
	texture<float,2,cudaReadModeElementType> tex_inputVelocity_mac_v;
	texture<float2,2,cudaReadModeElementType> tex_inputVelocity_regular;
	texture<float2,2,cudaReadModeElementType> tex_scalar; //temperature and density

	unsigned h_width;
	unsigned h_height;
	unsigned int h_steps;
	float h_voxelSize;
	float h_deltatt;
	
	__constant__ unsigned int d_width;
	__constant__ unsigned int d_height;
	__constant__ unsigned int d_steps;
	__constant__ float d_voxelSize;
	__constant__ float d_deltatt;
	
	/****************************************************************************************/
	
	__global__ 
	void ZQ_Cuda_Velocity_Advection_inRegular_outRegular_Kernel(float2 * d_output)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= d_width || y >= d_height)
			return;

		float2 pos = make_float2(x+0.5f,y+0.5f);
		float2 lastpos = pos;
		float2 velCoord = make_float2(pos.x/d_width,pos.y/d_height);
		float2 lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);


		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = velCoord;
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord = make_float2(pos.x/d_width,pos.y/d_height);

			lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);
			istep ++;
		} while (istep < d_steps);

		float2 out_coord = make_float2(lastpos.x/d_width,lastpos.y/d_height);
		float2 tempvel = tex2D(tex_inputVelocity_regular,out_coord.x,out_coord.y);
		d_output[y*d_width+x] = tempvel;
	}

	__global__ 
	void ZQ_Cuda_Velocity_Advection_inRegular_outMAC_u_Kernel(float * d_mac_u)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > d_width || y >= d_height)
			return;

		float2 pos = make_float2(x,y+0.5f);
		float2 lastpos = pos;
		float2 velCoord = make_float2(pos.x/d_width,pos.y/d_height);
		float2 lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);

		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = velCoord;
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord = make_float2(pos.x/d_width,pos.y/d_height);

			lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);
			istep ++;
		} while (istep < d_steps);

		float2 out_coord = make_float2(lastpos.x/d_width,lastpos.y/d_height);
		float2 tempvel = tex2D(tex_inputVelocity_regular,out_coord.x,out_coord.y);
		d_mac_u[y*(d_width+1)+x] = tempvel.x;
	}

	__global__
	void ZQ_Cuda_Velocity_Advection_inRegular_outMAC_v_Kernel(float * d_mac_v)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= d_width || y > d_height)
			return;

		float2 pos = make_float2(x+0.5f,y);
		float2 lastpos = pos;
		float2 velCoord = make_float2(pos.x/d_width,pos.y/d_height);
		float2 lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);

		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = velCoord;
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord = make_float2(pos.x/d_width,pos.y/d_height);

			lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);
			istep ++;
		} while (istep < d_steps);

		float2 out_coord = make_float2(lastpos.x/d_width,lastpos.y/d_height);
		float2 tempvel = tex2D(tex_inputVelocity_regular,out_coord.x,out_coord.y);
		d_mac_v[y*d_width+x] = tempvel.y;
	}

	__global__
	void ZQ_Cuda_Velocity_Advection_inMAC_outMAC_u_Kernel(float * d_mac_u)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x > d_width || y >= d_height)
			return;

		float2 pos = make_float2(x,y+0.5f);
		float2 lastpos = pos;
		float2 velCoord_u = make_float2((pos.x+0.5f)/(d_width+1),pos.y/d_height);
		float2 velCoord_v = make_float2(pos.x/d_width,(pos.y+0.5f)/(d_height+1));
		float2 lastvel = make_float2(
			tex2D(tex_velocity_MAC_u,velCoord_u.x,velCoord_u.y),
			tex2D(tex_velocity_MAC_v,velCoord_v.x,velCoord_v.y));

		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = make_float2(pos.x/d_width,pos.y/d_height);
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord_u = make_float2((pos.x+0.5f)/(d_width+1),pos.y/d_height);
			velCoord_v = make_float2(pos.x/d_width,(pos.y+0.5f)/(d_height+1));

			lastvel = make_float2(
				tex2D(tex_velocity_MAC_u,velCoord_u.x,velCoord_u.y),
				tex2D(tex_velocity_MAC_v,velCoord_v.x,velCoord_v.y));
			istep ++;
		} while (istep < d_steps);

		float2 out_coord_u = make_float2((lastpos.x+0.5f)/(d_width+1),lastpos.y/d_height);
		float tempvel = tex2D(tex_inputVelocity_mac_u,out_coord_u.x,out_coord_u.y);
		d_mac_u[y*(d_width+1)+x] = tempvel;
	}

	__global__
	void ZQ_Cuda_Velocity_Advection_inMAC_outMAC_v_Kernel(float * d_mac_v)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= d_width || y > d_height)
			return;

		float2 pos = make_float2(x+0.5f,y);
		float2 lastpos = pos;
		float2 velCoord_u = make_float2((pos.x+0.5f)/(d_width+1),pos.y/d_height);
		float2 velCoord_v = make_float2(pos.x/d_width,(pos.y+0.5f)/(d_height+1));
		float2 lastvel = make_float2(
			tex2D(tex_velocity_MAC_u,velCoord_u.x,velCoord_u.y),
			tex2D(tex_velocity_MAC_v,velCoord_v.x,velCoord_v.y));

		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = make_float2(pos.x/d_width,pos.y/d_height);
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord_u = make_float2((pos.x+0.5f)/(d_width+1),pos.y/d_height);
			velCoord_v = make_float2(pos.x/d_width,(pos.y+0.5f)/(d_height+1));

			lastvel = make_float2(
				tex2D(tex_velocity_MAC_u,velCoord_u.x,velCoord_u.y),
				tex2D(tex_velocity_MAC_v,velCoord_v.x,velCoord_v.y));
			istep ++;
		} while (istep < d_steps);

		float2 out_coord_v = make_float2(lastpos.x/d_width,(lastpos.y+0.5f)/(d_height+1));
		float tempvel = tex2D(tex_inputVelocity_mac_v,out_coord_v.x,out_coord_v.y);
		d_mac_v[y*d_width+x] = tempvel;
	}
		
	__global__ 
	void ZQ_Cuda_Scalar_Advection_Regular_Velocity_Kernel(float2* d_output)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= d_width || y >= d_height)
			return;

		float2 pos = make_float2(x+0.5f,y+0.5f);
		float2 lastpos = pos;
		float2 velCoord = make_float2(pos.x/d_width,pos.y/d_height);
		float2 lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);

		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = velCoord;
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord = make_float2(pos.x/d_width,pos.y/d_height);

			lastvel = tex2D(tex_velocity_regular,velCoord.x,velCoord.y);
			istep ++;
		} while (istep < d_steps);

		float2 out_coord = make_float2(lastpos.x/d_width,lastpos.y/d_height);
		float2 tempscalar = tex2D(tex_scalar,out_coord.x,out_coord.y);
		d_output[y*d_width+x] = tempscalar;
	}

	__global__ 
	void ZQ_Cuda_Scalar_Advection_MAC_Velocity_Kernel(float2* d_output)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= d_width || y >= d_height)
			return;

		float2 pos = make_float2(x+0.5f,y+0.5f);
		float2 lastpos = pos;
		float2 velCoord_u = make_float2((pos.x+0.5f)/(d_width+1),pos.y/d_height);
		float2 velCoord_v = make_float2(pos.x/d_width,(pos.y+0.5f)/(d_height+1));
		float2 lastvel = make_float2(
			tex2D(tex_velocity_MAC_u,velCoord_u.x,velCoord_u.y),
			tex2D(tex_velocity_MAC_v,velCoord_v.x,velCoord_v.y));

		unsigned int istep = 0;
		do 
		{
			float2 occupyCoord = make_float2(pos.x/d_width,pos.y/d_height);
			if(!(pos.x >= 0 && pos.x <= d_width && pos.y >= 0 && pos.y <= d_height))
				break;
			if(tex2D(tex_occupy,occupyCoord.x,occupyCoord.y) != 0)
				break;

			lastpos = pos;
			pos -= lastvel * d_deltatt / d_voxelSize;
			velCoord_u = make_float2((pos.x+0.5f)/(d_width+1),pos.y/d_height);
			velCoord_v = make_float2(pos.x/d_width,(pos.y+0.5f)/(d_height+1));

			lastvel = make_float2(
				tex2D(tex_velocity_MAC_u,velCoord_u.x,velCoord_u.y),
				tex2D(tex_velocity_MAC_v,velCoord_v.x,velCoord_v.y));
			istep ++;
		} while (istep < d_steps);

		float2 out_coord = make_float2(lastpos.x/d_width,lastpos.y/d_height);
		float2 tempscalar = tex2D(tex_scalar,out_coord.x,out_coord.y);
		d_output[y*d_width+x] = tempscalar;
	}

	extern "C"
	void ZQ_Cuda_Prepare_Advection2D(const unsigned int width, const unsigned int height, const float voxelSize, const unsigned int steps, const float deltatt)
	{
		h_width = width;
		h_height = height;
		h_steps = steps;
		h_voxelSize = voxelSize;
		h_deltatt = deltatt;

		checkCudaErrors( cudaMemcpyToSymbol(ZQ_CUDA_Advection2D::d_width,&width,sizeof(unsigned int)));
		checkCudaErrors( cudaMemcpyToSymbol(ZQ_CUDA_Advection2D::d_height,&height,sizeof(unsigned int)));
		checkCudaErrors( cudaMemcpyToSymbol(ZQ_CUDA_Advection2D::d_steps,&steps,sizeof(unsigned int)));
		checkCudaErrors( cudaMemcpyToSymbol(ZQ_CUDA_Advection2D::d_voxelSize,&voxelSize,sizeof(float)));
		checkCudaErrors( cudaMemcpyToSymbol(ZQ_CUDA_Advection2D::d_deltatt,&deltatt,sizeof(float)));	
		
		//int tmp = 0;
		//scanf("%d",&tmp);
		
		//checkCudaErrors( cudaMemcpyFromSymbol(&h_width,d_width,sizeof(unsigned int)));
		//checkCudaErrors( cudaMemcpyFromSymbol(&h_height,d_height,sizeof(unsigned int)));
		//checkCudaErrors( cudaMemcpyFromSymbol(&h_steps,d_steps,sizeof(unsigned int)));
		//checkCudaErrors( cudaMemcpyFromSymbol(&h_voxelSize,d_voxelSize,sizeof(float)));
		//checkCudaErrors( cudaMemcpyFromSymbol(&h_deltatt,d_deltatt,sizeof(float)));
		//printf("width = %d\n",h_width);
		//printf("height = %d\n",h_height);
		//printf("steps = %d\n",h_steps);
		//printf("voxelSize = %f\n",h_voxelSize);
		//printf("deltatt = %f\n",h_deltatt);
		
		//scanf("%d",&tmp);
			

	}


	extern "C"
	void ZQ_Cuda_Velocity_Advection2D_inRegular_outRegular(const float* velocity, const bool* occupy, const float* inVelocity, float* outVelocity)
	{
		tex_velocity_regular.normalized = true;
		tex_velocity_regular.filterMode = cudaFilterModeLinear;
		tex_velocity_regular.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_regular.addressMode[1] = cudaAddressModeClamp;
		
		tex_occupy.normalized = true;                      
		tex_occupy.filterMode = cudaFilterModePoint;     
		tex_occupy.addressMode[0] = cudaAddressModeClamp; 
		tex_occupy.addressMode[1] = cudaAddressModeClamp;

		tex_inputVelocity_regular.normalized = true;                      
		tex_inputVelocity_regular.filterMode = cudaFilterModeLinear;     
		tex_inputVelocity_regular.addressMode[0] = cudaAddressModeClamp; 
		tex_inputVelocity_regular.addressMode[1] = cudaAddressModeClamp;

		cudaChannelFormatDesc channelDescf2 = cudaCreateChannelDesc<float2>();
		cudaArray* velocity_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(velocity_array,0,0,velocity,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		cudaChannelFormatDesc channelDescb = cudaCreateChannelDesc<uchar1>();
		cudaArray* occupy_array = 0;
		checkCudaErrors( cudaMallocArray(&occupy_array,&channelDescb,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(occupy_array,0,0,occupy,sizeof(bool)*h_width*h_height,cudaMemcpyHostToDevice) );

		cudaArray* input_velocity_array = 0;
		checkCudaErrors( cudaMallocArray(&input_velocity_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(input_velocity_array,0,0,inVelocity,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaBindTextureToArray(tex_velocity_regular,velocity_array,channelDescf2) );
		checkCudaErrors( cudaBindTextureToArray(tex_occupy,occupy_array,channelDescb) );
		checkCudaErrors( cudaBindTextureToArray(tex_inputVelocity_regular,input_velocity_array,channelDescf2) );

		float2* d_output = 0;
		checkCudaErrors( cudaMalloc((void**)&d_output,sizeof(float)*h_width*h_height*2) );
		checkCudaErrors( cudaMemset(d_output,0,sizeof(float)*h_width*h_height*2) );
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((h_width+blockSize.x-1)/blockSize.x,(h_height+blockSize.y-1)/blockSize.y);
		ZQ_Cuda_Velocity_Advection_inRegular_outRegular_Kernel<<<gridSize,blockSize>>>(d_output);
		checkCudaErrors( cudaMemcpy(outVelocity,d_output,sizeof(float)*h_width*h_height*2,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(d_output) );
		d_output = 0;

		checkCudaErrors( cudaUnbindTexture(tex_velocity_regular) );
		checkCudaErrors( cudaUnbindTexture(tex_occupy) );
		checkCudaErrors( cudaUnbindTexture(tex_inputVelocity_regular) );
		checkCudaErrors( cudaFreeArray(velocity_array) );
		checkCudaErrors( cudaFreeArray(occupy_array) );
		checkCudaErrors( cudaFreeArray(input_velocity_array) );
		velocity_array = 0;
		occupy_array = 0;
		input_velocity_array = 0;
	}

	extern "C"
	void ZQ_Cuda_Velocity_Advection2D_inRegular_outMAC(const float* velocity, const bool* occupy, const float* inVelocity, float* out_mac_u, float* out_mac_v)
	{
		tex_velocity_regular.normalized = true;
		tex_velocity_regular.filterMode = cudaFilterModeLinear;
		tex_velocity_regular.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_regular.addressMode[1] = cudaAddressModeClamp;

		tex_occupy.normalized = true;                      
		tex_occupy.filterMode = cudaFilterModePoint;     
		tex_occupy.addressMode[0] = cudaAddressModeClamp; 
		tex_occupy.addressMode[1] = cudaAddressModeClamp;

		tex_inputVelocity_regular.normalized = true;                      
		tex_inputVelocity_regular.filterMode = cudaFilterModeLinear;     
		tex_inputVelocity_regular.addressMode[0] = cudaAddressModeClamp; 
		tex_inputVelocity_regular.addressMode[1] = cudaAddressModeClamp;

		cudaChannelFormatDesc channelDescf2 = cudaCreateChannelDesc<float2>();
		cudaArray* velocity_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(velocity_array,0,0,velocity,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		cudaChannelFormatDesc channelDescb = cudaCreateChannelDesc<uchar1>();
		cudaArray* occupy_array = 0;
		checkCudaErrors( cudaMallocArray(&occupy_array,&channelDescb,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(occupy_array,0,0,occupy,sizeof(bool)*h_width*h_height,cudaMemcpyHostToDevice) );

		cudaArray* input_velocity_array = 0;
		checkCudaErrors( cudaMallocArray(&input_velocity_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(input_velocity_array,0,0,inVelocity,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaBindTextureToArray(tex_velocity_regular,velocity_array,channelDescf2) );
		checkCudaErrors( cudaBindTextureToArray(tex_occupy,occupy_array,channelDescb) );
		checkCudaErrors( cudaBindTextureToArray(tex_inputVelocity_regular,input_velocity_array,channelDescf2) );

		float* d_out_mac_u = 0;
		float* d_out_mac_v = 0;
		checkCudaErrors( cudaMalloc((void**)&d_out_mac_u,sizeof(float)*(h_width+1)*h_height) );
		checkCudaErrors( cudaMemset(d_out_mac_u,0,sizeof(float)*(h_width+1)*h_height) );
		checkCudaErrors( cudaMalloc((void**)&d_out_mac_v,sizeof(float)*h_width*(h_height+1)) );
		checkCudaErrors( cudaMemset(d_out_mac_v,0,sizeof(float)*h_width*(h_height+1)) );
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize_u(((h_width+1)+blockSize.x-1)/blockSize.x,(h_height+blockSize.y-1)/blockSize.y);
		dim3 gridSize_v((h_width+blockSize.x-1)/blockSize.x,((h_height+1)+blockSize.y-1)/blockSize.y);
		ZQ_Cuda_Velocity_Advection_inRegular_outMAC_u_Kernel<<<gridSize_u,blockSize>>>(d_out_mac_u);
		ZQ_Cuda_Velocity_Advection_inRegular_outMAC_v_Kernel<<<gridSize_v,blockSize>>>(d_out_mac_v);
		checkCudaErrors( cudaMemcpy(out_mac_u,d_out_mac_u,sizeof(float)*(h_width+1)*h_height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(out_mac_v,d_out_mac_v,sizeof(float)*h_width*(h_height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(d_out_mac_u) );
		checkCudaErrors( cudaFree(d_out_mac_v) );
		d_out_mac_u = 0;
		d_out_mac_v = 0;

		checkCudaErrors( cudaUnbindTexture(tex_velocity_regular) );
		checkCudaErrors( cudaUnbindTexture(tex_occupy) );
		checkCudaErrors( cudaUnbindTexture(tex_inputVelocity_regular) );
		checkCudaErrors( cudaFreeArray(velocity_array) );
		checkCudaErrors( cudaFreeArray(occupy_array) );
		checkCudaErrors( cudaFreeArray(input_velocity_array) );
		velocity_array = 0;
		occupy_array = 0;
		input_velocity_array = 0;
	}

	extern "C"
	void ZQ_Cuda_Velocity_Advection2D_inMAC_outMAC(const float* vel_mac_u, const float* vel_mac_v, const bool* occupy, const float* in_mac_u, const float* in_mac_v,
						float* out_mac_u, float* out_mac_v)
	{
		tex_velocity_MAC_u.normalized = true;
		tex_velocity_MAC_u.filterMode = cudaFilterModeLinear;
		tex_velocity_MAC_u.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_MAC_u.addressMode[1] = cudaAddressModeClamp;

		tex_velocity_MAC_v.normalized = true;
		tex_velocity_MAC_v.filterMode = cudaFilterModeLinear;
		tex_velocity_MAC_v.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_MAC_v.addressMode[1] = cudaAddressModeClamp;


		tex_occupy.normalized = true;                      
		tex_occupy.filterMode = cudaFilterModePoint;     
		tex_occupy.addressMode[0] = cudaAddressModeClamp; 
		tex_occupy.addressMode[1] = cudaAddressModeClamp;

		tex_inputVelocity_mac_u.normalized = true;                      
		tex_inputVelocity_mac_u.filterMode = cudaFilterModeLinear;     
		tex_inputVelocity_mac_u.addressMode[0] = cudaAddressModeClamp; 
		tex_inputVelocity_mac_u.addressMode[1] = cudaAddressModeClamp;

		tex_inputVelocity_mac_v.normalized = true;                      
		tex_inputVelocity_mac_v.filterMode = cudaFilterModeLinear;     
		tex_inputVelocity_mac_v.addressMode[0] = cudaAddressModeClamp; 
		tex_inputVelocity_mac_v.addressMode[1] = cudaAddressModeClamp;

		cudaChannelFormatDesc channelDescf = cudaCreateChannelDesc<float>();
		cudaArray* velocity_u_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_u_array,&channelDescf,h_width+1,h_height) );
		checkCudaErrors( cudaMemcpyToArray(velocity_u_array,0,0,vel_mac_u,sizeof(float)*(h_width+1)*h_height,cudaMemcpyHostToDevice) );

		cudaArray* velocity_v_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_v_array,&channelDescf,h_width,h_height+1) );
		checkCudaErrors( cudaMemcpyToArray(velocity_v_array,0,0,vel_mac_v,sizeof(float)*h_width*(h_height+1),cudaMemcpyHostToDevice) );

		cudaChannelFormatDesc channelDescb = cudaCreateChannelDesc<uchar1>();
		cudaArray* occupy_array = 0;
		checkCudaErrors( cudaMallocArray(&occupy_array,&channelDescb,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(occupy_array,0,0,occupy,sizeof(bool)*h_width*h_height,cudaMemcpyHostToDevice) );

		cudaArray* input_velocity_u_array = 0;
		checkCudaErrors( cudaMallocArray(&input_velocity_u_array,&channelDescf,h_width+1,h_height) );
		checkCudaErrors( cudaMemcpyToArray(input_velocity_u_array,0,0,in_mac_u,sizeof(float)*(h_width+1)*h_height,cudaMemcpyHostToDevice) );

		cudaArray* input_velocity_v_array = 0;
		checkCudaErrors( cudaMallocArray(&input_velocity_v_array,&channelDescf,h_width,h_height+1) );
		checkCudaErrors( cudaMemcpyToArray(input_velocity_v_array,0,0,in_mac_v,sizeof(float)*h_width*(h_height+1),cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaBindTextureToArray(tex_velocity_MAC_u,velocity_u_array,channelDescf) );
		checkCudaErrors( cudaBindTextureToArray(tex_velocity_MAC_v,velocity_v_array,channelDescf) );
		checkCudaErrors( cudaBindTextureToArray(tex_occupy,occupy_array,channelDescb) );
		checkCudaErrors( cudaBindTextureToArray(tex_inputVelocity_mac_u,input_velocity_u_array,channelDescf) );
		checkCudaErrors( cudaBindTextureToArray(tex_inputVelocity_mac_v,input_velocity_v_array,channelDescf) );

		float* d_out_mac_u = 0;
		float* d_out_mac_v = 0;
		checkCudaErrors( cudaMalloc((void**)&d_out_mac_u,sizeof(float)*(h_width+1)*h_height) );
		checkCudaErrors( cudaMemset(d_out_mac_u,0,sizeof(float)*(h_width+1)*h_height) );
		checkCudaErrors( cudaMalloc((void**)&d_out_mac_v,sizeof(float)*h_width*(h_height+1)) );
		checkCudaErrors( cudaMemset(d_out_mac_v,0,sizeof(float)*h_width*(h_height+1)) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize_u(((h_width+1)+blockSize.x-1)/blockSize.x,(h_height+blockSize.y-1)/blockSize.y);
		dim3 gridSize_v((h_width+blockSize.x-1)/blockSize.x,((h_height+1)+blockSize.y-1)/blockSize.y);
		ZQ_Cuda_Velocity_Advection_inMAC_outMAC_u_Kernel<<<gridSize_u,blockSize>>>(d_out_mac_u);
		ZQ_Cuda_Velocity_Advection_inMAC_outMAC_v_Kernel<<<gridSize_v,blockSize>>>(d_out_mac_v);
		checkCudaErrors( cudaMemcpy(out_mac_u,d_out_mac_u,sizeof(float)*(h_width+1)*h_height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(out_mac_v,d_out_mac_v,sizeof(float)*h_width*(h_height+1),cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(d_out_mac_u) );
		checkCudaErrors( cudaFree(d_out_mac_v) );
		d_out_mac_u = 0;
		d_out_mac_v = 0;

		checkCudaErrors( cudaUnbindTexture(tex_velocity_MAC_u) );
		checkCudaErrors( cudaUnbindTexture(tex_velocity_MAC_v) );
		checkCudaErrors( cudaUnbindTexture(tex_occupy) );
		checkCudaErrors( cudaUnbindTexture(tex_inputVelocity_mac_u) );
		checkCudaErrors( cudaUnbindTexture(tex_inputVelocity_mac_v) );
		checkCudaErrors( cudaFreeArray(velocity_u_array) );
		checkCudaErrors( cudaFreeArray(velocity_v_array) );
		checkCudaErrors( cudaFreeArray(occupy_array) );
		checkCudaErrors( cudaFreeArray(input_velocity_u_array) );
		checkCudaErrors( cudaFreeArray(input_velocity_v_array) );
		velocity_u_array = 0;
		velocity_v_array = 0;
		occupy_array = 0;
		input_velocity_u_array = 0;
		input_velocity_v_array = 0;
	}

	extern "C"
	void ZQ_Cuda_Scalar_Advection2D_Regular_Velocity(const float* velocity, const bool* occupy, const float* input, float* output)
	{
		tex_velocity_regular.normalized = true;
		tex_velocity_regular.filterMode = cudaFilterModeLinear;
		tex_velocity_regular.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_regular.addressMode[1] = cudaAddressModeClamp;


		tex_occupy.normalized = true;                      
		tex_occupy.filterMode = cudaFilterModePoint;     
		tex_occupy.addressMode[0] = cudaAddressModeClamp; 
		tex_occupy.addressMode[1] = cudaAddressModeClamp;

		tex_scalar.normalized = true;
		tex_scalar.filterMode = cudaFilterModeLinear;
		tex_scalar.addressMode[0] = cudaAddressModeClamp;
		tex_scalar.addressMode[1] = cudaAddressModeClamp;

		cudaChannelFormatDesc channelDescf2 = cudaCreateChannelDesc<float2>();
		cudaArray* velocity_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(velocity_array,0,0,velocity,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		cudaChannelFormatDesc channelDescb = cudaCreateChannelDesc<uchar1>();
		cudaArray* occupy_array = 0;
		checkCudaErrors( cudaMallocArray(&occupy_array,&channelDescb,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(occupy_array,0,0,occupy,sizeof(bool)*h_width*h_height,cudaMemcpyHostToDevice) );

		cudaArray* scalar_array = 0;
		checkCudaErrors( cudaMallocArray(&scalar_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(scalar_array,0,0,input,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaBindTextureToArray(tex_velocity_regular,velocity_array,channelDescf2) );
		checkCudaErrors( cudaBindTextureToArray(tex_occupy,occupy_array,channelDescb) );
		checkCudaErrors( cudaBindTextureToArray(tex_scalar,scalar_array,channelDescf2) );

		float2* d_output = 0;
		checkCudaErrors( cudaMalloc((void**)&d_output,sizeof(float)*h_width*h_height*2) );
		checkCudaErrors( cudaMemset(d_output,0,sizeof(float)*h_width*h_height*2) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((h_width+blockSize.x-1)/blockSize.x,(h_height+blockSize.y-1)/blockSize.y);
		ZQ_Cuda_Scalar_Advection_Regular_Velocity_Kernel<<<gridSize,blockSize>>>(d_output);
		checkCudaErrors( cudaMemcpy(output,d_output,sizeof(float)*h_width*h_height*2,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(d_output) );
		d_output = 0;

		checkCudaErrors( cudaUnbindTexture(tex_velocity_regular) );
		checkCudaErrors( cudaUnbindTexture(tex_occupy) );
		checkCudaErrors( cudaUnbindTexture(tex_scalar) );
		checkCudaErrors( cudaFreeArray(velocity_array) );
		checkCudaErrors( cudaFreeArray(occupy_array) );
		checkCudaErrors( cudaFreeArray(scalar_array) );
		velocity_array = 0;
		occupy_array = 0;
		scalar_array = 0;
	}

	extern "C"
	void ZQ_Cuda_Scalar_Advection2D_MAC_Velocity(const float* vel_mac_u, const float* vel_mac_v, const bool* occupy, const float* input, float* output)
	{
		tex_velocity_MAC_u.normalized = true;
		tex_velocity_MAC_u.filterMode = cudaFilterModeLinear;
		tex_velocity_MAC_u.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_MAC_u.addressMode[1] = cudaAddressModeClamp;

		tex_velocity_MAC_v.normalized = true;
		tex_velocity_MAC_v.filterMode = cudaFilterModeLinear;
		tex_velocity_MAC_v.addressMode[0] = cudaAddressModeClamp;
		tex_velocity_MAC_v.addressMode[1] = cudaAddressModeClamp;


		tex_occupy.normalized = true;                      
		tex_occupy.filterMode = cudaFilterModePoint;     
		tex_occupy.addressMode[0] = cudaAddressModeClamp; 
		tex_occupy.addressMode[1] = cudaAddressModeClamp;

		tex_scalar.normalized = true;
		tex_scalar.filterMode = cudaFilterModeLinear;
		tex_scalar.addressMode[0] = cudaAddressModeClamp;
		tex_scalar.addressMode[1] = cudaAddressModeClamp;

		cudaChannelFormatDesc channelDescf = cudaCreateChannelDesc<float>();
		cudaArray* velocity_u_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_u_array,&channelDescf,h_width+1,h_height) );
		checkCudaErrors( cudaMemcpyToArray(velocity_u_array,0,0,vel_mac_u,sizeof(float)*(h_width+1)*h_height,cudaMemcpyHostToDevice) );

		cudaArray* velocity_v_array = 0;
		checkCudaErrors( cudaMallocArray(&velocity_v_array,&channelDescf,h_width,h_height+1) );
		checkCudaErrors( cudaMemcpyToArray(velocity_v_array,0,0,vel_mac_v,sizeof(float)*h_width*(h_height+1),cudaMemcpyHostToDevice) );

		cudaChannelFormatDesc channelDescb = cudaCreateChannelDesc<uchar1>();
		cudaArray* occupy_array = 0;
		checkCudaErrors( cudaMallocArray(&occupy_array,&channelDescb,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(occupy_array,0,0,occupy,sizeof(bool)*h_width*h_height,cudaMemcpyHostToDevice) );

		
		cudaChannelFormatDesc channelDescf2 = cudaCreateChannelDesc<float2>();

		cudaArray* scalar_array = 0;
		checkCudaErrors( cudaMallocArray(&scalar_array,&channelDescf2,h_width,h_height) );
		checkCudaErrors( cudaMemcpyToArray(scalar_array,0,0,input,sizeof(float)*h_width*h_height*2,cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaBindTextureToArray(tex_velocity_MAC_u,velocity_u_array,channelDescf) );
		checkCudaErrors( cudaBindTextureToArray(tex_velocity_MAC_v,velocity_v_array,channelDescf) );
		checkCudaErrors( cudaBindTextureToArray(tex_occupy,occupy_array,channelDescb) );
		checkCudaErrors( cudaBindTextureToArray(tex_scalar,scalar_array,channelDescf2) );

		float2* d_output = 0;
		checkCudaErrors( cudaMalloc((void**)&d_output,sizeof(float)*h_width*h_height*2) );
		checkCudaErrors( cudaMemset(d_output,0,sizeof(float)*h_width*h_height*2) );
		
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((h_width+blockSize.x-1)/blockSize.x,(h_height+blockSize.y-1)/blockSize.y);
		ZQ_Cuda_Scalar_Advection_MAC_Velocity_Kernel<<<gridSize,blockSize>>>(d_output);
		checkCudaErrors( cudaMemcpy(output,d_output,sizeof(float)*h_width*h_height*2,cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaFree(d_output) );
		d_output = 0;

		checkCudaErrors( cudaUnbindTexture(tex_velocity_MAC_u) );
		checkCudaErrors( cudaUnbindTexture(tex_velocity_MAC_v) );
		checkCudaErrors( cudaUnbindTexture(tex_occupy) );
		checkCudaErrors( cudaUnbindTexture(tex_scalar) );
		checkCudaErrors( cudaFreeArray(velocity_u_array) );
		checkCudaErrors( cudaFreeArray(velocity_v_array) );
		checkCudaErrors( cudaFreeArray(occupy_array) );
		checkCudaErrors( cudaFreeArray(scalar_array) );
		velocity_u_array = 0;
		velocity_v_array = 0;
		occupy_array = 0;
		scalar_array = 0;
	}
	
}


#endif