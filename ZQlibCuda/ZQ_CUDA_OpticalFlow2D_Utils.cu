#ifndef _ZQ_CUDA_OPTICAL_FLOW_2D_UTILS_CU_
#define _ZQ_CUDA_OPTICAL_FLOW_2D_UTILS_CU_

#include "ZQ_CUDA_OpticalFlow2D_Utils.cuh"
#include "ZQ_CUDA_ImageProcessing2D.cuh"
#include "ZQ_CUDA_PoissonSolver2D.cuh"

namespace ZQ_CUDA_OpticalFlow2D
{
	/****************   Base Kernels  **********************************/
	__global__
	void compute_psi_data_Kernel(float* psi_data, const float* imdx, const float* imdy, const float* imdt, 
							const float* du, const float* dv, const float eps, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		int offset = y*width+x;

		float value = 0;
		for(int i = 0;i < nChannels;i++)
		{
			float tmp = (imdt[offset*nChannels+i]+imdx[offset*nChannels+i]*du[offset]+imdy[offset*nChannels+i]*dv[offset]);
			value += tmp*tmp;
		}

		psi_data[offset] = 0.5/sqrt(value+eps);
	}

	__global__
	void compute_psi_smooth_Kernel(float* psi_smooth, const float* u, const float* v, const float eps, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		int offset = y*width+x;
		
		float ux = (x < width-1) ? (u[offset+1]-u[offset]) : 0;
		float uy = (y < height-1) ? (u[offset+width]-u[offset]) : 0;
		float vx = (x < width-1) ? (v[offset+1]-v[offset]) : 0;
		float vy = (y < height-1) ? (v[offset+width]-v[offset]) : 0;

		psi_smooth[offset] = 0.5/sqrt(ux*ux+uy*uy+vx*vx+vy*vy+eps);

	}
	
	__global__
	void compute_psi_u_v_Kernel(float* psi_u, float* psi_v, const float* u, const float* v, const float eps, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		int offset = y*width+x;
		
		psi_u[offset] = 0.5/sqrt(u[offset]*u[offset]+eps);
		psi_v[offset] = 0.5/sqrt(v[offset]*v[offset]+eps);
	}

	__global__
	void compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_Kernel(float* imdxdx, float* imdxdy, float* imdydy, float* imdtdx, float* imdtdy, 
															const float* imdx, const float* imdy, const float* imdt,
															const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		int offset = y*width+x;

		imdxdx[offset] = 0;
		imdxdy[offset] = 0;
		imdydy[offset] = 0;
		imdtdx[offset] = 0;
		imdtdy[offset] = 0;

		for(int c = 0; c < nChannels;c++)
		{
			imdxdx[offset] += imdx[offset*nChannels+c]*imdx[offset*nChannels+c];
			imdxdy[offset] += imdx[offset*nChannels+c]*imdy[offset*nChannels+c];
			imdydy[offset] += imdy[offset*nChannels+c]*imdy[offset*nChannels+c];
			imdtdx[offset] += imdt[offset*nChannels+c]*imdx[offset*nChannels+c];
			imdtdy[offset] += imdt[offset*nChannels+c]*imdy[offset*nChannels+c];
		}
		imdxdx[offset] /= nChannels;
		imdxdy[offset] /= nChannels;
		imdydy[offset] /= nChannels;
		imdtdx[offset] /= nChannels;
		imdtdy[offset] /= nChannels;
	}

	__global__
	void compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_withpsidata_Kernel(float* imdxdx, float* imdxdy, float* imdydy, 
														  float* imdtdx, float* imdtdy, const float* imdx, const float* imdy, const float* imdt,
														  const float* psi_data, const int width, const int height, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		int offset = y*width+x;

		imdxdx[offset] = 0;
		imdxdy[offset] = 0;
		imdydy[offset] = 0;
		imdtdx[offset] = 0;
		imdtdy[offset] = 0;

		for(int c = 0; c < nChannels;c++)
		{
			imdxdx[offset] += imdx[offset*nChannels+c]*imdx[offset*nChannels+c];
			imdxdy[offset] += imdx[offset*nChannels+c]*imdy[offset*nChannels+c];
			imdydy[offset] += imdy[offset*nChannels+c]*imdy[offset*nChannels+c];
			imdtdx[offset] += imdt[offset*nChannels+c]*imdx[offset*nChannels+c];
			imdtdy[offset] += imdt[offset*nChannels+c]*imdy[offset*nChannels+c];
		}
		imdxdx[offset] *= psi_data[offset]/nChannels;
		imdxdy[offset] *= psi_data[offset]/nChannels;
		imdydy[offset] *= psi_data[offset]/nChannels;
		imdtdx[offset] *= psi_data[offset]/nChannels;
		imdtdy[offset] *= psi_data[offset]/nChannels;
	}
	
		
	__global__
	void Laplacian_withpsismooth_Kernel(float* output, const float* input,const float* psi_smooth, const int width, const int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;

		float value = 0;
		float in_x,in_x_,in_y,in_y_;

		in_x = (x < width-1) ? (input[offset+1] - input[offset]) : 0 ;
		in_x_ = (x > 0) ? (input[offset] - input[offset-1]) : 0;
		value += (x > 0) ? (psi_smooth[offset]*in_x - psi_smooth[offset-1]*in_x_) : 0;
		
		in_y = (y < height-1) ? (input[offset+width] - input[offset]) : 0;
		in_y_ = (y > 0) ? (input[offset] - input[offset-width]) : 0;
		value += (y > 0) ? (psi_smooth[offset]*in_y - psi_smooth[offset-width]*in_y_) : 0;

		output[offset] = value;
	}
	
	
	__global__
	void OpticalFlow_L2_RedBlack_Kernel(float* du, float* dv, const float* u, const float* v, const float* imdxdx, const float* imdxdy, const float* imdydy, 
										const float* imdtdx, const float* imdtdy, const float* laplace_u, const float* laplace_v, const int width, const int height, 
										const float alpha, const float beta, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redKernel ? 1 : 0))
			return;


		int i = y;
		int j = x;

		int offset = i * width + j;
		float sigma1 = 0, sigma2 = 0, coeff = 0;
		float _weight;

		if(j > 0)
		{
			_weight = 1;
			sigma1 += _weight*du[offset-1];
			sigma2 += _weight*dv[offset-1];
			coeff += _weight;

		}
		if(j < width-1)
		{
			_weight = 1;
			sigma1 += _weight*du[offset+1];
			sigma2 += _weight*dv[offset+1];
			coeff   += _weight;
		}
		if(i > 0)
		{
			_weight = 1;
			sigma1 += _weight*du[offset-width];
			sigma2 += _weight*dv[offset-width];
			coeff   += _weight;
		}
		if(i < height-1)
		{
			_weight = 1;
			sigma1  += _weight*du[offset+width];
			sigma2  += _weight*dv[offset+width];
			coeff   += _weight;
		}
		sigma1 *= alpha;
		sigma2 *= alpha;
		coeff *= alpha;
		// compute u
		sigma1 += alpha*laplace_u[offset] - imdtdx[offset] - imdxdy[offset]*dv[offset] - beta*u[offset];
		float coeff1 = coeff + imdxdx[offset] + beta;
		du[offset] = (1-omega)*du[offset] + omega/coeff1*sigma1;
		// compute v
		sigma2 += alpha*laplace_v[offset] - imdtdy[offset] - imdxdy[offset]*du[offset] - beta*v[offset];
		float coeff2 = coeff + imdydy[offset] + beta;
		dv[offset] = (1-omega)*dv[offset] + omega/coeff2*sigma2;

	}
		
	__global__
	void OpticalFlow_L1_RedBlack_Kernel(float* du, float* dv, const float* u, const float* v, const float* imdxdx, const float* imdxdy, const float* imdydy, const float* imdtdx, const float* imdtdy,
										const float* laplace_u, const float* laplace_v,const float* psi_smooth, const float* psi_u, const float* psi_v,
										const int width, const int height, const float alpha, const float beta, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redKernel ? 1 : 0))
			return;


		int i = y;
		int j = x;

		int offset = i * width + j;
		float sigma1 = 0, sigma2 = 0, coeff = 0;
		float _weight;

		if(j > 0)
		{
			_weight = psi_smooth[offset-1];
			sigma1 += _weight*du[offset-1];
			sigma2 += _weight*dv[offset-1];
			coeff += _weight;

		}
		if(j < width-1)
		{
			_weight = psi_smooth[offset];
			sigma1 += _weight*du[offset+1];
			sigma2 += _weight*dv[offset+1];
			coeff   += _weight;
		}
		if(i > 0)
		{
			_weight = psi_smooth[offset-width];
			sigma1 += _weight*du[offset-width];
			sigma2 += _weight*dv[offset-width];
			coeff   += _weight;
		}
		if(i < height-1)
		{
			_weight = psi_smooth[offset];
			sigma1  += _weight*du[offset+width];
			sigma2  += _weight*dv[offset+width];
			coeff   += _weight;
		}
		sigma1 *= alpha;
		sigma2 *= alpha;
		coeff *= alpha;
		// compute u
		sigma1 += alpha*laplace_u[offset] - imdtdx[offset] - imdxdy[offset]*dv[offset] - beta*psi_u[offset]*u[offset];
		float coeff1 = coeff + imdxdx[offset] + beta*psi_u[offset];
		du[offset] = (1-omega)*du[offset] + omega/coeff1*sigma1;
		// compute v
		sigma2 += alpha*laplace_v[offset] - imdtdy[offset] - imdxdy[offset]*du[offset] - beta*psi_v[offset]*v[offset];
		float coeff2 = coeff + imdydy[offset] + beta*psi_v[offset];
		dv[offset] = (1-omega)*dv[offset] + omega/coeff2*sigma2;

	}
	
	/******************************* for ADMM  ********************************************/

	__global__
	void proximalF_RedBlack_Kernel(float* du, float* dv, const float* imdxdx, const float* imdxdy, const float* imdydy, const float* imdtdx, const float* imdtdy,
									   const float* laplace_u, const float* laplace_v, const float* u, const float* z_u, const float* v, const float* z_v,
									   const int width, const int height, const float alpha, const float beta, const float lambda, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		if(rest == (redKernel ? 1 : 0))
			return;


		int i = y;
		int j = x;

		int offset = i * width + j;
		float sigma1 = 0, sigma2 = 0, coeff = 0;
		float _weight;

		if(j > 0)
		{
			_weight = 1;
			sigma1 += _weight*du[offset-1];
			sigma2 += _weight*dv[offset-1];
			coeff += _weight;

		}
		if(j < width-1)
		{
			_weight = 1;
			sigma1 += _weight*du[offset+1];
			sigma2 += _weight*dv[offset+1];
			coeff   += _weight;
		}
		if(i > 0)
		{
			_weight = 1;
			sigma1 += _weight*du[offset-width];
			sigma2 += _weight*dv[offset-width];
			coeff   += _weight;
		}
		if(i < height-1)
		{
			_weight = 1;
			sigma1  += _weight*du[offset+width];
			sigma2  += _weight*dv[offset+width];
			coeff   += _weight;
		}
		sigma1 *= alpha;
		sigma2 *= alpha;
		coeff *= alpha;
		// compute u
		sigma1 += alpha*laplace_u[offset] - imdtdx[offset] - imdxdy[offset]*dv[offset] - beta*u[offset] - 0.5*lambda*(u[offset] - z_u[offset]);
		float coeff1 = coeff + imdxdx[offset] + beta + 0.5*lambda;
		du[offset] = (1-omega)*du[offset] + omega/coeff1*sigma1;
		// compute v
		sigma2 += alpha*laplace_v[offset] - imdtdy[offset] - imdxdy[offset]*du[offset] - beta*v[offset] - 0.5*lambda*(v[offset] - z_v[offset]);
		float coeff2 = coeff + imdydy[offset] + beta + 0.5*lambda;
		dv[offset] = (1-omega)*dv[offset] + omega/coeff2*sigma2;

	}
	
	__global__
	void proximal_F2_Kernel(float* u, float* v, const float* z_u, const float* z_v, const float* warpU, const float* warpV, 
							const int width, const int height, const float gama, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;
		
		u[offset] = (gama*warpU[offset]+0.5*lambda*z_u[offset])/(gama+0.5*lambda);
		v[offset] = (gama*warpV[offset]+0.5*lambda*z_v[offset])/(gama+0.5*lambda);

	}
	
	__global__
	void compute_z_u_z_v_for_proximal_F1_Kernel(float* z_u, float* z_v, const float* u_for_F1, const float* v_for_F1, 
												const float* u_for_q1, const float* v_for_q1, const int width, const int height, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;
		z_u[offset] = u_for_F1[offset] - 1.0/lambda*u_for_q1[offset];
		z_v[offset] = v_for_F1[offset] - 1.0/lambda*v_for_q1[offset];
	}

	__global__
	void compute_z_u_z_v_for_proximal_F2_Kernel(float* z_u, float* z_v, const float* u_for_F2, const float* v_for_F2, 
												const float* u_for_q2, const float* v_for_q2, const int width, const int height, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;
		z_u[offset] = u_for_F2[offset] - 1.0/lambda*u_for_q2[offset];
		z_v[offset] = v_for_F2[offset] - 1.0/lambda*v_for_q2[offset];
	}

	__global__
	void compute_z_u_z_v_for_proximal_G_Kernel(float* z_u, float* z_v, const float* u_for_F1, const float* v_for_F1,
											   const float* u_for_F2, const float* v_for_F2, const float* u_for_q1, const float* v_for_q1,
											   const float* u_for_q2, const float* v_for_q2, const int width, const int height, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;
		z_u[offset] = 0.5*(u_for_F1[offset] + 1.0/lambda*u_for_q1[offset] + u_for_F2[offset] + 1.0/lambda*u_for_q2[offset]);
		z_v[offset] = 0.5*(v_for_F1[offset] + 1.0/lambda*v_for_q1[offset] + v_for_F2[offset] + 1.0/lambda*v_for_q2[offset]);
	}
	
	__global__
	void update_u_v_for_q1_q2_Kernel(float* u_for_q1, float* v_for_q1, float* u_for_q2, float* v_for_q2,
									 const float* u_for_F1, const float* v_for_F1, const float* u_for_F2, const float* v_for_F2,
									 const float* u_for_G, const float* v_for_G, const int width, const int height, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int offset = y*width+x;
		u_for_q1[offset] += lambda*(u_for_F1[offset] - u_for_G[offset]);
		v_for_q1[offset] += lambda*(v_for_F1[offset] - v_for_G[offset]);
		u_for_q2[offset] += lambda*(u_for_F2[offset] - u_for_G[offset]);
		v_for_q2[offset] += lambda*(v_for_F2[offset] - v_for_G[offset]);
	}
	
	/****************************************************************************************************/
	
	void cu_Compute_z_u_z_v_for_proximal_F1(float* z_u, float* z_v, const float* u_for_F1, const float* v_for_F1, 
										 const float* u_for_q1, const float* v_for_q1, const int width, const int height, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		compute_z_u_z_v_for_proximal_F1_Kernel<<<gridSize,blockSize>>>(z_u,z_v,u_for_F1,v_for_F1,u_for_q1,v_for_q1,width,height,lambda);
	}

	void cu_Compute_z_u_z_v_for_proximal_F2(float* z_u, float* z_v, const float* u_for_F2, const float* v_for_F2, 
										 const float* u_for_q2, const float* v_for_q2, const int width, const int height, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		compute_z_u_z_v_for_proximal_F2_Kernel<<<gridSize,blockSize>>>(z_u,z_v,u_for_F2,v_for_F2,u_for_q2,v_for_q2,width,height,lambda);
	}

	void cu_Compute_z_u_z_v_for_proximal_G(float* z_u, float* z_v, const float* u_for_F1, const float* v_for_F1,
										const float* u_for_F2, const float* v_for_F2, const float* u_for_q1, const float* v_for_q1,
										const float* u_for_q2, const float* v_for_q2, const int width, const int height, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		compute_z_u_z_v_for_proximal_G_Kernel<<<gridSize,blockSize>>>(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,
			u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,lambda);
	}

	void cu_Update_u_v_for_q1_q2(float* u_for_q1, float* v_for_q1, float* u_for_q2, float* v_for_q2,
									 const float* u_for_F1, const float* v_for_F1, const float* u_for_F2, const float* v_for_F2,
									 const float* u_for_G, const float* v_for_G, const int width, const int height, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		update_u_v_for_q1_q2_Kernel<<<gridSize,blockSize>>>(u_for_q1,v_for_q1,u_for_q2,v_for_q2,
			u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,lambda);
	}

	void cu_GetDerivatives(float* imdx, float* imdy, float* imdt, const float* Im1, const float* Im2, const int width, const int height, const int nChannels)
	{

		float* tmpBuf = 0;
		float* im1 = 0;
		float* im2 = 0;
		checkCudaErrors( cudaMalloc((void**)&tmpBuf, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&im1, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&im2, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(tmpBuf,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(im1,0,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(im2,0,sizeof(float)*width*height*nChannels) );


		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		ZQ_CUDA_ImageProcessing2D::Imfilter_h_Gaussian_Kernel<<<gridSize,blockSize>>>(tmpBuf,Im1,width,height,nChannels);
		ZQ_CUDA_ImageProcessing2D::Imfilter_v_Gaussian_Kernel<<<gridSize,blockSize>>>(im1,tmpBuf,width,height,nChannels);

		ZQ_CUDA_ImageProcessing2D::Imfilter_h_Gaussian_Kernel<<<gridSize,blockSize>>>(tmpBuf,Im2,width,height,nChannels);
		ZQ_CUDA_ImageProcessing2D::Imfilter_v_Gaussian_Kernel<<<gridSize,blockSize>>>(im2,tmpBuf,width,height,nChannels);

		/* tmpBuf = im1*0.4 + im2*0.6 */
		ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmpBuf,im1,0.4,im2,0.6,width,height,nChannels);

		/* imdx = \partial_x {tmpBuf} */
		ZQ_CUDA_ImageProcessing2D::Derivative_x_Advanced_Kernel<<<gridSize,blockSize>>>(imdx,tmpBuf,width,height,nChannels);

		/* imdy = \partial_y {tmpBuf} */
		ZQ_CUDA_ImageProcessing2D::Derivative_y_Advanced_Kernel<<<gridSize,blockSize>>>(imdy,tmpBuf,width,height,nChannels);

		ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(imdt,im2,1,im1,-1,width,height,nChannels);

		checkCudaErrors( cudaFree(tmpBuf) );
		checkCudaErrors( cudaFree(im1) );
		checkCudaErrors( cudaFree(im2) );
		tmpBuf = 0;
		im1 = 0;
		im2 = 0;
	}
	
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L2(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter)
	{
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );



		int nOuterFPIterations = nOuterFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );

			/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
			compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,width,height,nChannels);


			/* laplace u, v */
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,1);


			// set omega
			float omega = 1.0f;
			float alpha2 = alpha*alpha;
			float beta2 = beta*beta;
		
			/* red - black solver begin */
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,
					width,height,alpha2,beta2,omega,true);
				OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,
					width,height,alpha2,beta2,omega,false);
			}
			/* red - black solver end */
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

		
		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
		
	}
	
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L2_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter)
	{
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );



		int nOuterFPIterations = nOuterFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Occupy_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,occupy,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );

			/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
			compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,width,height,nChannels);


			/* laplace u, v */
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,1);


			// set omega
			float omega = 1.0f;
			float alpha2 = alpha*alpha;
			float beta2 = beta*beta;
		
			/* red - black solver begin */
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,
					width,height,alpha2,beta2,omega,true);
				OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,
					width,height,alpha2,beta2,omega,false);
			}
			/* red - black solver end */
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Occupy_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,occupy,u,v,width,height,nChannels);

		
		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
		
	}
	
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter,const int nSORIter)
	{
	
		float eps = optical_flow_L1_eps;
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* psi_data = 0;
		float* psi_smooth = 0;
		float* psi_u = 0;
		float* psi_v = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&psi_data,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&psi_smooth,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&psi_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&psi_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(psi_data,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(psi_smooth,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(psi_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(psi_v,0,sizeof(float)*width*height) );



		int nOuterFPIterations = nOuterFPIter;
		int nInnerFPIterations = nInnerFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{
			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
			
			for(int inner_it = 0; inner_it < nInnerFPIterations;inner_it++)
			{
				/* compute psi_data*/
				compute_psi_data_Kernel<<<gridSize,blockSize>>>(psi_data,imdx,imdy,imdt,du,dv,eps,width,height,nChannels);

				/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
				compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_withpsidata_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,psi_data,width,height,nChannels);


				/*compute psi_smooth*/
				ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
				ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
				
				compute_psi_smooth_Kernel<<<gridSize,blockSize>>>(psi_smooth,u,v,eps,width,height);
				
				compute_psi_u_v_Kernel<<<gridSize,blockSize>>>(psi_u,psi_v,u,v,eps,width,height);
				
				ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,-1,width,height,1);
				ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,-1,width,height,1);

				/* laplace u, v with psi_smooth */
				Laplacian_withpsismooth_Kernel<<<gridSize,blockSize>>>(laplace_u,u,psi_smooth,width,height);
				Laplacian_withpsismooth_Kernel<<<gridSize,blockSize>>>(laplace_v,v,psi_smooth,width,height);


				// set omega
				float omega = 1.0f;
				
			
				/* red - black solver begin */
				for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
				{
					OpticalFlow_L1_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,psi_smooth,
						psi_u,psi_v,width,height,alpha,beta,omega,true);
					OpticalFlow_L1_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,psi_smooth,
						psi_u,psi_v,width,height,alpha,beta,omega,false);
				}
				/* red - black solver end */
			}
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);
		
		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(psi_data) );
		checkCudaErrors( cudaFree(psi_smooth) );
		checkCudaErrors( cudaFree(psi_u) );
		checkCudaErrors( cudaFree(psi_v) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
		psi_data = 0;
		psi_smooth = 0;
		psi_u = 0;
		psi_v = 0;
	}
	
	void cu_OpticalFlow_DL1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter)
	{
		float eps = optical_flow_L1_eps;
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* psi_data = 0;


		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&psi_data,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(psi_data,0,sizeof(float)*width*height) );



		int nOuterFPIterations = nOuterFPIter;
		int nInnerFPIterations = nInnerFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
			
			for(int inner_it = 0; inner_it < nInnerFPIterations;inner_it++)
			{

				/* compute psi_data*/
				compute_psi_data_Kernel<<<gridSize,blockSize>>>(psi_data,imdx,imdy,imdt,du,dv,eps,width,height,nChannels);

				/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
				compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_withpsidata_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,psi_data,
																								width,height,nChannels);


				/* laplace u, v with psi_smooth */
				ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,1);
				ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,1);


				// set omega
				float omega = 1.0;
				float alpha2 = alpha*alpha;
				float beta2 = beta*beta;
				
			
				/* red - black solver begin */
				for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
				{
					OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,
						width,height,alpha2,beta2,omega,true);
					OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,u,v,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,
						width,height,alpha2,beta2,omega,false);
				}
				/* red - black solver end */
			}
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(psi_data) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
		psi_data = 0;
	}
	
	void cu_Proximal_F1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* z_u, const float* z_v, 
				const int width, const int height, const int nChannels, const float alpha, const float beta, const float lambda, const int nOuterFPIter, const int nSORIter)
	{
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );



		/* ProximalF(z_u,z_v,\lambda) = minimize_{u,v} \int {|I_2(x+u,y+v)-I_1(x,y)|^2} + \alpha^2 \int {|\nabla u|^2 + |\nabla v|^2} + \beta^2 \int {|u|^2 + |v|^2} + \lambda \int {|u-z_u|^2 + |v-z_v|^2} 
		*
		* The Euler-Lagrange equation is:
		*  I_t I_x + \beta^2 u + \lambda(u-z_u) = \alpha^2 \Delta u 
		*  I_t I_y + \beta^2 v + \lambda(v-z_v) = \alpha^2 \Delta v
		*/

		int nOuterFPIterations = nOuterFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );

			/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
			compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,width,height,nChannels);


			/* laplace u, v */
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,1);


			// set omega
			float omega = 1.0;
			float alpha2 = alpha*alpha;
			float beta2 = beta*beta;


			/* red - black solver begin */
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,u,z_u,v,z_v,
					width,height,alpha2,beta2,lambda,omega,true);
				proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,u,z_u,v,z_v,
					width,height,alpha2,beta2,lambda,omega,false);
			}
			/* red - black solver end */
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
	}
	
	void cu_Proximal_F1_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* z_u, const float* z_v, 
				const int width, const int height, const int nChannels, const float alpha, const float beta, const float lambda, const int nOuterFPIter, const int nSORIter)
	{
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );



		/* ProximalF(z_u,z_v,\lambda) = minimize_{u,v} \int {|I_2(x+u,y+v)-I_1(x,y)|^2} + \alpha^2 \int {|\nabla u|^2 + |\nabla v|^2} + \beta^2 \int {|u|^2 + |v|^2} + \lambda \int {|u-z_u|^2 + |v-z_v|^2} 
		*
		* The Euler-Lagrange equation is:
		*  I_t I_x + \beta^2 u + \lambda(u-z_u) = \alpha^2 \Delta u 
		*  I_t I_y + \beta^2 v + \lambda(v-z_v) = \alpha^2 \Delta v
		*/

		int nOuterFPIterations = nOuterFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Occupy_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,occupy,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );

			/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
			compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,width,height,nChannels);


			/* laplace u, v */
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,1);


			// set omega
			float omega = 1.0;
			float alpha2 = alpha*alpha;
			float beta2 = beta*beta;


			/* red - black solver begin */
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,u,z_u,v,z_v,
					width,height,alpha2,beta2,lambda,omega,true);
				proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,u,z_u,v,z_v,
					width,height,alpha2,beta2,lambda,omega,false);
			}
			/* red - black solver end */
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Occupy_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,occupy,u,v,width,height,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
	}
	
	void cu_Proximal_F1_DL1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* z_u, const float* z_v, 
					const int width, const int height, const int nChannels, const float alpha, const float beta, const float lambda, 
					const int nOuterFPIter, const int nInnerFPIter, const int nSORIter)
	{
		
		float eps = optical_flow_L1_eps;
		float* du = 0;
		float* dv = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdydy = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* psi_data = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&psi_data, sizeof(float)*width*height) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(psi_data,0, sizeof(float)*width*height) );



		/* ProximalF(z_u,z_v,\lambda) = minimize_{u,v} \int {|I_2(x+u,y+v)-I_1(x,y)|^2} + \alpha^2 \int {|\nabla u|^2 + |\nabla v|^2} + \beta^2 \int {|u|^2 + |v|^2} + \lambda \int {|u-z_u|^2 + |v-z_v|^2} 
		*
		* The Euler-Lagrange equation is:
		*  I_t I_x + \beta^2 u + \lambda(u-z_u) = \alpha^2 \Delta u 
		*  I_t I_y + \beta^2 v + \lambda(v-z_v) = \alpha^2 \Delta v
		*/

		int nOuterFPIterations = nOuterFPIter;
		int nInnerFPIterations = nInnerFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image bicubic */
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdt,Im1,warpIm2,width,height,nChannels);

			/* reset du, dv */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height) );
			
			for(int inner_it = 0; inner_it < nInnerFPIterations; inner_it++)
			{
				/*compute psi_data*/
				compute_psi_data_Kernel<<<gridSize,blockSize>>>(psi_data,imdx,imdy,imdt,du,dv,eps,width,height,nChannels);

				/* compute imdxdx, imdxdy, imdydy, imdtdx, imdtdy */
				compute_imdxdx_imdxdy_imdydy_imdtdx_imtdy_withpsidata_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdydy,imdtdx,imdtdy,imdx,imdy,imdt,psi_data,
																									width,height,nChannels);


				/* laplace u, v */
				ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,1);
				ZQ_CUDA_ImageProcessing2D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,1);


				// set omega
				float omega = 1.0;
				float alpha2 = alpha*alpha;
				float beta2 = beta*beta;


				/* red - black solver begin */
				for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
				{
					proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,u,z_u,v,z_v,
						width,height,alpha2,beta2,lambda,omega,true);
					proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,imdxdx,imdxdy,imdydy,imdtdx,imdtdy,laplace_u,laplace_v,u,z_u,v,z_v,
						width,height,alpha2,beta2,lambda,omega,false);
				}
				/* red - black solver end */
			}
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,width,height,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(psi_data) );
		
		du = 0;
		dv = 0;
		laplace_u = 0;
		laplace_v = 0;
		imdx = 0;
		imdy = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdydy = 0;
		imdtdx = 0;
		imdtdy = 0;
		psi_data = 0;

	}
	
	void cu_Proximal_F2_first(float* u, float* v, const float* z_u, const float* z_v, const float* next_u, const float* next_v, 
				const int width, const int height, const float gama, const float lambda, const int nFPIter, const int nPoissonIter)
	{
		int nOuterFPIterations = nFPIter;
		int nPoissonIterations = nPoissonIter;

		float* warpU = 0;
		float* warpV = 0;

		checkCudaErrors( cudaMalloc((void**)&warpU,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpV,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpU,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpV,0,sizeof(float)*width*height) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		for(int out_it = 0;out_it < nOuterFPIterations;out_it++)
		{
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpU,u,next_u,u,v,width,height,1);
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpV,v,next_v,u,v,width,height,1);

			ZQ_CUDA_PoissonSolver2D::cu_SolveOpenPoissonRedBlack_Regular(warpU,warpV,width,height,nPoissonIterations);

			proximal_F2_Kernel<<<gridSize,blockSize>>>(u,v,z_u,z_v,warpU,warpV,width,height,gama,lambda);
		}

		checkCudaErrors( cudaFree(warpU) );
		checkCudaErrors( cudaFree(warpV) );
		warpU = 0;
		warpV = 0;
	}


	void cu_Proximal_F2_middle(float* u, float* v, const float* z_u, const float* z_v, const float* pre_u, const float* pre_v, const float* next_u, const float* next_v, 
						const int width, const int height, const float gama, const float lambda, const int nFPIter, const int nPoissonIter)
	{
		int nOuterFPIterations = nFPIter;
		int nPoissonIterations = nPoissonIter;

		float* warpU_pre = 0;
		float* warpV_pre = 0;
		float* warpU_nex = 0;
		float* warpV_nex = 0;
		float* tmp_u = 0;
		float* tmp_v = 0;

		checkCudaErrors( cudaMalloc((void**)&warpU_pre,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpV_pre,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpU_nex,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpV_nex,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpU_pre,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpV_pre,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpU_nex,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpV_nex,0,sizeof(float)*width*height) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		for(int out_it = 0;out_it < nOuterFPIterations;out_it++)
		{
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpU_nex,u,next_u,u,v,width,height,1);
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpV_nex,v,next_v,u,v,width,height,1);

			checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height) );
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,1);

			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpU_pre,u,pre_u,tmp_u,tmp_v,width,height,1);
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpV_pre,v,pre_v,tmp_u,tmp_v,width,height,1);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmp_u,warpU_pre,0.5,warpU_nex,0.5,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmp_v,warpV_pre,0.5,warpV_nex,0.5,width,height,1);


			ZQ_CUDA_PoissonSolver2D::cu_SolveOpenPoissonRedBlack_Regular(tmp_u,tmp_v,width,height,nPoissonIterations);

			proximal_F2_Kernel<<<gridSize,blockSize>>>(u,v,z_u,z_v,tmp_u,tmp_v,width,height,2*gama,lambda);
		}

		checkCudaErrors( cudaFree(warpU_pre) );
		checkCudaErrors( cudaFree(warpV_pre) );
		checkCudaErrors( cudaFree(warpU_nex) );
		checkCudaErrors( cudaFree(warpV_nex) );
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		warpU_pre = 0;
		warpV_pre = 0;
		warpU_nex = 0;
		warpV_nex = 0;
		tmp_u = 0;
		tmp_v = 0;
	}

	void cu_Proximal_F2_last(float* u, float* v, const float* z_u, const float* z_v, const float* pre_u, const float* pre_v,
							const int width, const int height, const float gama, const float lambda, const int nFPIter, const int nPoissonIter)
	{
		int nOuterFPIterations = nFPIter;
		int nPoissonIterations = nPoissonIter;

		float* warpU = 0;
		float* warpV = 0;
		float* tmp_u = 0;
		float* tmp_v = 0;

		checkCudaErrors( cudaMalloc((void**)&warpU,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpV,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpU,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(warpV,0,sizeof(float)*width*height) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		for(int out_it = 0;out_it < nOuterFPIterations;out_it++)
		{
			checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height) );
			checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height) );
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,1);

			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpU,u,pre_u,tmp_u,tmp_v,width,height,1);
			ZQ_CUDA_ImageProcessing2D::WarpImage_Bicubic_Kernel<<<gridSize,blockSize>>>(warpV,v,pre_v,tmp_u,tmp_v,width,height,1);

			ZQ_CUDA_PoissonSolver2D::cu_SolveOpenPoissonRedBlack_Regular(warpU,warpV,width,height,nPoissonIterations);

			proximal_F2_Kernel<<<gridSize,blockSize>>>(u,v,z_u,z_v,warpU,warpV,width,height,gama,lambda);
		}

		checkCudaErrors( cudaFree(warpU) );
		checkCudaErrors( cudaFree(warpV) );
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		warpU = 0;
		warpV = 0;
		tmp_u = 0;
		tmp_v = 0;
	}


	void cu_Proximal_G(float* u, float* v, const float* z_u, const float* z_v, const int width, const int height, const int nPoissonIter)
	{
		checkCudaErrors( cudaMemcpy(u,z_u,sizeof(float)*width*height, cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v,z_v,sizeof(float)*width*height, cudaMemcpyDeviceToDevice) );

		int nPoissonIterations = nPoissonIter;

		ZQ_CUDA_PoissonSolver2D::cu_SolveOpenPoissonRedBlack_Regular(u,v,width,height,nPoissonIterations);
	}
	
	void cu_OpticalFlow_ADMM(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter)
	{
		float* u_for_F = u;
		float* v_for_F = v;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q = 0;
		float* v_for_q = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_G,1,u_for_q,-1.0,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_G,1,v_for_q,-1.0,width,height,1);

			
			cu_Proximal_F1(u_for_F,v_for_F,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_F,1,u_for_q,1.0,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_F,1,v_for_q,1.0,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_F,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_G,-1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_F,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_G,-1,width,height,1);

		}
		
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q) );
		checkCudaErrors( cudaFree(v_for_q) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		 u_for_F = 0;
		 v_for_F = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q = 0;
		 v_for_q = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter)
	{
		float* u_for_F = u;
		float* v_for_F = v;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q = 0;
		float* v_for_q = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_G,1,u_for_q,-1.0,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_G,1,v_for_q,-1.0,width,height,1);

			
			cu_Proximal_F1_Occupy(u_for_F,v_for_F,warpIm2,Im1,Im2,occupy,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_F,1,u_for_q,1.0,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_F,1,v_for_q,1.0,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_F,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_G,-1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_F,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_G,-1,width,height,1);

		}
		
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q) );
		checkCudaErrors( cudaFree(v_for_q) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		 u_for_F = 0;
		 v_for_F = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q = 0;
		 v_for_q = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_DL1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, 
						const int nSORIter, const int nPoissonIter)
	{
		float* u_for_F = u;
		float* v_for_F = v;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q = 0;
		float* v_for_q = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_G,1,u_for_q,-1.0,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_G,1,v_for_q,-1.0,width,height,1);

			
			cu_Proximal_F1_DL1(u_for_F,v_for_F,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_F,1,u_for_q,1.0,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_F,1,v_for_q,1.0,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_F,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_G,-1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_F,1,width,height,1);
			ZQ_CUDA_ImageProcessing2D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_G,-1,width,height,1);

		}
		
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q) );
		checkCudaErrors( cudaFree(v_for_q) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		 u_for_F = 0;
		 v_for_F = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q = 0;
		 v_for_q = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	
	void cu_OpticalFlow_ADMM_First(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		float new_gamma = gamma*alpha*alpha;

		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1(u_for_F1,v_for_F1,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_first(u_for_F2,v_for_F2,z_u,z_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);
		
			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}
		
		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		 u_for_F1 = 0;
		 v_for_F1 = 0;
		 u_for_F2 = 0;
		 v_for_F2 = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q1 = 0;
		 v_for_q1 = 0;
		 u_for_q2 = 0;
		 v_for_q2 = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_First_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* next_u, const float* next_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		float new_gamma = gamma*alpha*alpha;

		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1_Occupy(u_for_F1,v_for_F1,warpIm2,Im1,Im2,occupy,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_first(u_for_F2,v_for_F2,z_u,z_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);
		
			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}
		
		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		 u_for_F1 = 0;
		 v_for_F1 = 0;
		 u_for_F2 = 0;
		 v_for_F2 = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q1 = 0;
		 v_for_q1 = 0;
		 u_for_q2 = 0;
		 v_for_q2 = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_DL1_First(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		float new_gamma = gamma*alpha*alpha;

		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1_DL1(u_for_F1,v_for_F1,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_first(u_for_F2,v_for_F2,z_u,z_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);
		
			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);
			
			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}
		
		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		 u_for_F1 = 0;
		 v_for_F1 = 0;
		 u_for_F2 = 0;
		 v_for_F2 = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 u_for_q1 = 0;
		 v_for_q1 = 0;
		 u_for_q2 = 0;
		 v_for_q2 = 0;
		 z_u = 0;
		 z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_Middle(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		

		float new_gamma = gamma*alpha*alpha;


		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);
			
			cu_Proximal_F1(u_for_F1,v_for_F1,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_middle(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_Middle_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );


		

		float new_gamma = gamma*alpha*alpha;


		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);
			
			cu_Proximal_F1_Occupy(u_for_F1,v_for_F1,warpIm2,Im1,Im2,occupy,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_middle(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_DL1_Middle(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		float new_gamma = gamma*alpha*alpha;


		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);
			
			cu_Proximal_F1_DL1(u_for_F1,v_for_F1,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_middle(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,next_u,next_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		

		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
	}
	
	void cu_OpticalFlow_ADMM_Last(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		float new_gamma = gamma*alpha*alpha;
		
		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1(u_for_F1,v_for_F1,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_last(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
			
	}
	
	void cu_OpticalFlow_ADMM_Last_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* pre_u, const float* pre_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		float new_gamma = gamma*alpha*alpha;
		
		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1_Occupy(u_for_F1,v_for_F1,warpIm2,Im1,Im2,occupy,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_last(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
			
	}
	
	void cu_OpticalFlow_ADMM_DL1_Last(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height) );

		float new_gamma = gamma*alpha*alpha;
		
		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_for_proximal_F1(z_u,z_v,u_for_G,v_for_G,u_for_q1,v_for_q1,width,height,1);

			cu_Proximal_F1_DL1(u_for_F1,v_for_F1,warpIm2,Im1,Im2,z_u,z_v,width,height,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			cu_Compute_z_u_z_v_for_proximal_F2(z_u,z_v,u_for_G,v_for_G,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_F2_last(u_for_F2,v_for_F2,z_u,z_v,pre_u,pre_v,width,height,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_for_proximal_G(z_u,z_v,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_q1,v_for_q1,u_for_q2,v_for_q2,width,height,1);

			cu_Proximal_G(u_for_G,v_for_G,z_u,z_v,width,height,nPoissonIter);

			cu_Update_u_v_for_q1_q2(u_for_q1,v_for_q1,u_for_q2,v_for_q2,u_for_F1,v_for_F1,u_for_F2,v_for_F2,u_for_G,v_for_G,width,height,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		
		u_for_F1 = 0;
		v_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		z_u = 0;
		z_v = 0;
			
	}
	
	/***********************************************************************/
	
	extern "C"
	void InitDevice2D(const int deviceid)
	{
		int num_devices = 0;
		checkCudaErrors(cudaGetDeviceCount(&num_devices));
		int cur_device = deviceid;
		if(deviceid < 0 || deviceid >= num_devices)
		{
			cur_device = 0;
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, cur_device);
			printf("use the Device ID:\t%d\n", cur_device);
			printf("Device Name is used:\t%s\n", properties.name );
      	}
      	checkCudaErrors(cudaSetDevice(cur_device));
	}
	
	extern "C"
	float OpticalFlow2D_L2(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_L2(u_d,v_d,warpIm2_d,Im1_d,Im2_d,width,height,nChannels,alpha,beta,nOuterFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float OpticalFlow2D_L2_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const int width, const int height, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_L2_Occupy(u_d,v_d,warpIm2_d,Im1_d,Im2_d,occupy_d,width,height,nChannels,alpha,beta,nOuterFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float OpticalFlow2D_L1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter,const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_L1(u_d,v_d,warpIm2_d,Im1_d,Im2_d,width,height,nChannels,alpha,beta,nOuterFPIter,nInnerFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;

	}
	
	extern "C"
	float OpticalFlow2D_DL1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_DL1(u_d,v_d,warpIm2_d,Im1_d,Im2_d,width,height,nChannels,alpha,beta,nOuterFPIter,nInnerFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;

	}
	
	extern "C"
	float OpticalFlow2D_ADMM(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels,
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM(u_d,v_d,warpIm2_d,Im1_d,Im2_d,width,height,nChannels,alpha,beta,lambda,ADMMIter,nOuterFPIter,nSORIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float OpticalFlow2D_ADMM_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const int width, const int height, const int nChannels,
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;
		float* warpIm2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Occupy(u_d,v_d,warpIm2_d,Im1_d,Im2_d,occupy_d,width,height,nChannels,alpha,beta,lambda,ADMMIter,nOuterFPIter,nSORIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		occupy_d = 0;
		warpIm2_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float OpticalFlow2D_ADMM_DL1(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int nChannels,
				const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, 
				const int nSORIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_d = 0;
		float* v_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );

		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1(u_d,v_d,warpIm2_d,Im1_d,Im2_d,width,height,nChannels,alpha,beta,lambda,ADMMIter,nOuterFPIter,nInnerFPIter,nSORIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float OpticalFlow2D_ADMM_First(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_First(u_d,v_d,warpIm2_d,Im1_d,Im2_d,next_u_d,next_v_d,width,height,nChannels,alpha,beta,gamma,lambda,
									ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	extern "C"
	float OpticalFlow2D_ADMM_First_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* next_u, const float* next_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;
		float* warpIm2_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_First_Occupy(u_d,v_d,warpIm2_d,Im1_d,Im2_d,occupy_d,next_u_d,next_v_d,width,height,nChannels,alpha,beta,gamma,lambda,
									ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		occupy_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}

	extern "C"
	float OpticalFlow2D_ADMM_DL1_First(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1_First(u_d,v_d,warpIm2_d,Im1_d,Im2_d,next_u_d,next_v_d,width,height,nChannels,alpha,beta,gamma,lambda,
							ADMMIter,nOuterFPIter,nInnerFPIter, nSORIter,nWarpFPIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	
	extern "C"
	float OpticalFlow2D_ADMM_Middle(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Middle(u_d,v_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,next_u_d,next_v_d,width,height,nChannels,
									alpha,beta,gamma,lambda,ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);
	
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		u_d = 0;
		v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float OpticalFlow2D_ADMM_Middle_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;
		float* warpIm2_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Middle_Occupy(u_d,v_d,warpIm2_d,Im1_d,Im2_d,occupy_d,pre_u_d,pre_v_d,next_u_d,next_v_d,width,height,nChannels,
									alpha,beta,gamma,lambda,ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);
	
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		
		Im1_d = 0;
		Im2_d = 0;
		occupy_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		u_d = 0;
		v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}

	extern "C"
	float OpticalFlow2D_ADMM_DL1_Middle(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							   const float* next_u, const float* next_v, const int width, const int height, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		float* next_u_d = 0;
		float* next_v_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1_Middle(u_d,v_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,next_u_d,next_v_d,width,height,nChannels,
									alpha,beta,gamma,lambda,ADMMIter,nOuterFPIter,nInnerFPIter,nSORIter,nWarpFPIter,nPoissonIter);
	
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		u_d = 0;
		v_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	
	extern "C"
	float OpticalFlow2D_ADMM_Last(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		
		
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Last(u_d,v_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,width,height,nChannels,alpha,beta,gamma,lambda,
							ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);
		
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	extern "C"
	float OpticalFlow2D_ADMM_Last_Occupy(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* occupy, const float* pre_u, const float* pre_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* occupy_d = 0;
		float* warpIm2_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		
		
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&occupy_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(occupy_d,occupy,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Last_Occupy(u_d,v_d,warpIm2_d,Im1_d,Im2_d,occupy_d,pre_u_d,pre_v_d,width,height,nChannels,alpha,beta,gamma,lambda,
							ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);
		
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(occupy_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		occupy_d = 0;
		warpIm2_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	extern "C"
	float OpticalFlow2D_ADMM_DL1_Last(float* u, float* v, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v,
							  const int width, const int height, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		
		
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1_Last(u_d,v_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,width,height,nChannels,alpha,beta,gamma,lambda,
									ADMMIter,nOuterFPIter,nSORIter,nInnerFPIter,nWarpFPIter,nPoissonIter);
		
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		
		u_d = 0;
		v_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
}

#endif