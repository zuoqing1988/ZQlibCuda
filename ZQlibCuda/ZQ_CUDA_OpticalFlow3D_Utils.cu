#ifndef _ZQ_CUDA_OPTICAL_FLOW_3D_UTILS_CU_
#define _ZQ_CUDA_OPTICAL_FLOW_3D_UTILS_CU_

#include "ZQ_CUDA_OpticalFlow3D_Utils.cuh"
#include "ZQ_CUDA_ImageProcessing3D.cuh"
#include "ZQ_CUDA_PoissonSolver3D.cuh"

namespace ZQ_CUDA_OpticalFlow3D
{	
	/****************   Base Kernels  **********************************/
	__global__
	void compute_psi_data_Kernel(float* psi_data, const float* imdx, const float* imdy, const float* imdz, const float* imdt, 
							const float* du, const float* dv, const float* dw, const float eps, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float value = 0;
			for(int i = 0;i < nChannels;i++)
			{
				float tmp = (imdt[offset*nChannels+i]+imdx[offset*nChannels+i]*du[offset]+imdy[offset*nChannels+i]*dv[offset]+imdz[offset*nChannels+i]*dw[offset]);
				value += tmp*tmp;
			}

			psi_data[offset] = 0.5/sqrt(value+eps);
		}
	}
	
	__global__
	void compute_psi_smooth_Kernel(float* psi_smooth, const float* u, const float* v, const float* w, const float eps, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			
			float ux = (x < width-1) ? (u[offset+1]-u[offset]) : 0;
			float uy = (y < height-1) ? (u[offset+width]-u[offset]) : 0;
			float uz = (z < depth-1) ? (u[offset+height*width]-u[offset]) : 0;
			float vx = (x < width-1) ? (v[offset+1]-v[offset]) : 0;
			float vy = (y < height-1) ? (v[offset+width]-v[offset]) : 0;
			float vz = (z < depth-1) ? (v[offset+height*width]-v[offset]) : 0;
			float wx = (x < width-1) ? (w[offset+1]-w[offset]) : 0;
			float wy = (y < height-1) ? (w[offset+width]-w[offset]) : 0;
			float wz = (z < depth-1) ? (w[offset+height*width]-w[offset]) : 0;

			psi_smooth[offset] = 0.5/sqrt(ux*ux+uy*uy+uz*uz+vx*vx+vy*vy+vz*vz+wx*wx+wy*wy+wz*wz+eps);
		}
	}
	
	__global__
	void compute_psi_u_v_w_Kernel(float* psi_u, float* psi_v, float* psi_w, const float* u, const float* v, const float* w, const float eps, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{		
			int offset = z*height*width+y*width+x;
			
			psi_u[offset] = 0.5f/sqrt(u[offset]*u[offset]+eps);
			psi_v[offset] = 0.5f/sqrt(v[offset]*v[offset]+eps);
			psi_w[offset] = 0.5f/sqrt(w[offset]*w[offset]+eps);
		}
	}
	
	__global__
	void compute_imdxdx_imdtdx_Kernel(float* imdxdx, float* imdxdy, float* imdxdz, float* imdydy, float* imdydz, float* imdzdz, 
									float* imdtdx, float* imdtdy, float* imdtdz, const float* imdx, const float* imdy, const float* imdz, const float* imdt,
									const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			imdxdx[offset] = 0;
			imdxdy[offset] = 0;
			imdxdz[offset] = 0;
			imdydy[offset] = 0;
			imdydz[offset] = 0;
			imdzdz[offset] = 0;
			imdtdx[offset] = 0;
			imdtdy[offset] = 0;
			imdtdz[offset] = 0;

			for(int c = 0; c < nChannels;c++)
			{
				imdxdx[offset] += imdx[offset*nChannels+c]*imdx[offset*nChannels+c];
				imdxdy[offset] += imdx[offset*nChannels+c]*imdy[offset*nChannels+c];
				imdxdz[offset] += imdx[offset*nChannels+c]*imdz[offset*nChannels+c];
				imdydy[offset] += imdy[offset*nChannels+c]*imdy[offset*nChannels+c];
				imdydz[offset] += imdy[offset*nChannels+c]*imdz[offset*nChannels+c];
				imdzdz[offset] += imdz[offset*nChannels+c]*imdz[offset*nChannels+c];
				imdtdx[offset] += imdt[offset*nChannels+c]*imdx[offset*nChannels+c];
				imdtdy[offset] += imdt[offset*nChannels+c]*imdy[offset*nChannels+c];
				imdtdz[offset] += imdt[offset*nChannels+c]*imdz[offset*nChannels+c];
			}
			imdxdx[offset] /= nChannels;
			imdxdy[offset] /= nChannels;
			imdxdz[offset] /= nChannels;
			imdydy[offset] /= nChannels;
			imdydz[offset] /= nChannels;
			imdzdz[offset] /= nChannels;
			imdtdx[offset] /= nChannels;
			imdtdy[offset] /= nChannels;
			imdtdz[offset] /= nChannels;
		}
	}
	
	__global__
	void compute_imdxdx_imdtdx_withpsidata_Kernel(float* imdxdx, float* imdxdy, float* imdxdz, float* imdydy, float* imdydz, float* imdzdz,
												float* imdtdx, float* imdtdy, float* imdtdz, const float* imdx, const float* imdy, const float* imdz, const float* imdt,
												const float* psi_data, const int width, const int height, const int depth, const int nChannels)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;
		
		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			imdxdx[offset] = 0;
			imdxdy[offset] = 0;
			imdxdz[offset] = 0;
			imdydy[offset] = 0;
			imdydz[offset] = 0;
			imdzdz[offset] = 0;
			imdtdx[offset] = 0;
			imdtdy[offset] = 0;
			imdtdz[offset] = 0;

			for(int c = 0; c < nChannels;c++)
			{
				imdxdx[offset] += imdx[offset*nChannels+c]*imdx[offset*nChannels+c];
				imdxdy[offset] += imdx[offset*nChannels+c]*imdy[offset*nChannels+c];
				imdxdz[offset] += imdx[offset*nChannels+c]*imdz[offset*nChannels+c];
				imdydy[offset] += imdy[offset*nChannels+c]*imdy[offset*nChannels+c];
				imdydz[offset] += imdy[offset*nChannels+c]*imdz[offset*nChannels+c];
				imdzdz[offset] += imdz[offset*nChannels+c]*imdz[offset*nChannels+c];
				imdtdx[offset] += imdt[offset*nChannels+c]*imdx[offset*nChannels+c];
				imdtdy[offset] += imdt[offset*nChannels+c]*imdy[offset*nChannels+c];
				imdtdz[offset] += imdt[offset*nChannels+c]*imdz[offset*nChannels+c];
			}
			imdxdx[offset] *= psi_data[offset]/nChannels;
			imdxdy[offset] *= psi_data[offset]/nChannels;
			imdxdz[offset] *= psi_data[offset]/nChannels;
			imdydy[offset] *= psi_data[offset]/nChannels;
			imdydz[offset] *= psi_data[offset]/nChannels;
			imdzdz[offset] *= psi_data[offset]/nChannels;
			imdtdx[offset] *= psi_data[offset]/nChannels;
			imdtdy[offset] *= psi_data[offset]/nChannels;
			imdtdz[offset] *= psi_data[offset]/nChannels;
		}
	}
	
	__global__
	void Laplacian_withpsismooth_Kernel(float* output, const float* input,const float* psi_smooth, const int width, const int height, const int depth)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;

			float value = 0;
			float in_x,in_x_,in_y,in_y_,in_z,in_z_;

			in_x = (x < width-1) ? (input[offset+1] - input[offset]) : 0 ;
			in_x_ = (x > 0) ? (input[offset] - input[offset-1]) : 0;
			value += (x > 0) ? (psi_smooth[offset]*in_x - psi_smooth[offset-1]*in_x_) : 0;
			
			in_y = (y < height-1) ? (input[offset+width] - input[offset]) : 0;
			in_y_ = (y > 0) ? (input[offset] - input[offset-width]) : 0;
			value += (y > 0) ? (psi_smooth[offset]*in_y - psi_smooth[offset-width]*in_y_) : 0;
			
			in_z = (z < depth-1) ? (input[offset+height*width] - input[offset]) : 0;
			in_z_ = (z > 0) ? (input[offset] - input[offset-height*width]) : 0;
			value += (z > 0) ? (psi_smooth[offset]*in_z - psi_smooth[offset-height*width]*in_z_) : 0;

			output[offset] = value;
		}
	}
	
	__global__
	void OpticalFlow_L2_RedBlack_Kernel(float* du, float* dv, float* dw, const float* u, const float* v, const float* w,
									const float* imdxdx, const float* imdxdy, const float* imdxdz, const float* imdydy, const float* imdydz, const float* imdzdz,
									const float* imdtdx, const float* imdtdy, const float* imdtdz, const float* laplace_u, const float* laplace_v, const float* laplace_w,
									const int width, const int height, const int depth,	const float alpha, const float beta, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redKernel ? rest : (1-rest);

		for(int z = start;z < depth;z += 2)
		{
			int offset = z*height*width+y*width+x;
			float sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
			float _weight;

			if(x > 0)
			{
				_weight = 1;
				sigma1 += _weight*du[offset-1];
				sigma2 += _weight*dv[offset-1];
				sigma3 += _weight*dw[offset-1];
				coeff  += _weight;

			}
			if(x < width-1)
			{
				_weight = 1;
				sigma1 += _weight*du[offset+1];
				sigma2 += _weight*dv[offset+1];
				sigma3 += _weight*dw[offset+1];
				coeff  += _weight;
			}
			if(y > 0)
			{
				_weight = 1;
				sigma1 += _weight*du[offset-width];
				sigma2 += _weight*dv[offset-width];
				sigma3 += _weight*dw[offset-width];
				coeff  += _weight;
			}
			if(y < height-1)
			{
				_weight = 1;
				sigma1 += _weight*du[offset+width];
				sigma2 += _weight*dv[offset+width];
				sigma3 += _weight*dw[offset+width];
				coeff  += _weight;
			}
			if(z > 0)
			{
				_weight = 1;
				sigma1 += _weight*du[offset-height*width];
				sigma2 += _weight*dv[offset-height*width];
				sigma3 += _weight*dw[offset-height*width];
				coeff  += _weight;
			}
			if(z < depth-1)
			{
				_weight = 1;
				sigma1 += _weight*du[offset+height*width];
				sigma2 += _weight*dv[offset+height*width];
				sigma3 += _weight*dw[offset+height*width];
				coeff  += _weight;
			}
			sigma1 *= alpha;
			sigma2 *= alpha;
			sigma3 *= alpha;
			coeff *= alpha;
			// compute u
			sigma1 += alpha*laplace_u[offset] - imdtdx[offset] - imdxdy[offset]*dv[offset] - imdxdz[offset]*dw[offset] - beta*u[offset];
			float coeff1 = coeff + imdxdx[offset] + beta;
			du[offset] = (1-omega)*du[offset] + omega/coeff1*sigma1;
			// compute v
			sigma2 += alpha*laplace_v[offset] - imdtdy[offset] - imdxdy[offset]*du[offset] - imdydz[offset]*dw[offset] - beta*v[offset];
			float coeff2 = coeff + imdydy[offset] + beta;
			dv[offset] = (1-omega)*dv[offset] + omega/coeff2*sigma2;
			// compute w
			sigma3 += alpha*laplace_w[offset] - imdtdz[offset] - imdxdz[offset]*du[offset] - imdydz[offset]*dv[offset] - beta*w[offset];
			float coeff3 = coeff + imdzdz[offset] + beta;
			dw[offset] = (1-omega)*dw[offset] + omega/coeff3*sigma3;
		}
	}
	
	__global__
	void OpticalFlow_L1_RedBlack_Kernel(float* du, float* dv, float* dw, const float* u, const float* v, const float* w,
								const float* imdxdx, const float* imdxdy, const float* imdxdz, const float* imdydy, const float* imdydz, const float* imdzdz,
								const float* imdtdx, const float* imdtdy, const float* imdtdz, const float* laplace_u, const float* laplace_v, const float* laplace_w,
								const float* psi_smooth, const float* psi_u, const float* psi_v, const float* psi_w,
								const int width, const int height, const int depth, const float alpha, const float beta, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redKernel ? rest : (1-rest);
	
		for(int z = start;z < depth;z += 2)
		{
			int offset = z*height*width+y*width+x;
			float sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
			float _weight;

			if(x > 0)
			{
				_weight = psi_smooth[offset-1];
				sigma1 += _weight*du[offset-1];
				sigma2 += _weight*dv[offset-1];
				sigma3 += _weight*dw[offset-1];
				coeff  += _weight;
			}
			if(x < width-1)
			{
				_weight = psi_smooth[offset];
				sigma1 += _weight*du[offset+1];
				sigma2 += _weight*dv[offset+1];
				sigma3 += _weight*dw[offset+1];
				coeff  += _weight;
			}
			if(y > 0)
			{
				_weight = psi_smooth[offset-width];
				sigma1 += _weight*du[offset-width];
				sigma2 += _weight*dv[offset-width];
				sigma3 += _weight*dw[offset-width];
				coeff  += _weight;
			}
			if(y < height-1)
			{
				_weight = psi_smooth[offset];
				sigma1 += _weight*du[offset+width];
				sigma2 += _weight*dv[offset+width];
				sigma3 += _weight*dw[offset+width];
				coeff  += _weight;
			}
			if(z > 0)
			{
				_weight = psi_smooth[offset-height*width];
				sigma1 += _weight*du[offset-height*width];
				sigma2 += _weight*dv[offset-height*width];
				sigma3 += _weight*dw[offset-height*width];
				coeff  += _weight;
			}
			if(z < depth-1)
			{
				_weight = psi_smooth[offset+height*width];
				sigma1 += _weight*du[offset+height*width];
				sigma2 += _weight*dv[offset+height*width];
				sigma3 += _weight*dw[offset+height*width];
				coeff  += _weight;
			}
			sigma1 *= alpha;
			sigma2 *= alpha;
			sigma3 *= alpha;
			coeff *= alpha;
			// compute u
			sigma1 += alpha*laplace_u[offset] - imdtdx[offset] - imdxdy[offset]*dv[offset] - imdxdz[offset]*dw[offset] - beta*psi_u[offset]*u[offset];
			float coeff1 = coeff + imdxdx[offset] + beta*psi_u[offset];
			du[offset] = (1-omega)*du[offset] + omega/coeff1*sigma1;
			// compute v
			sigma2 += alpha*laplace_v[offset] - imdtdy[offset] - imdxdy[offset]*du[offset] - imdydz[offset]*dw[offset] - beta*psi_v[offset]*v[offset];
			float coeff2 = coeff + imdydy[offset] + beta*psi_v[offset];
			dv[offset] = (1-omega)*dv[offset] + omega/coeff2*sigma2;
			// compute w
			sigma3 += alpha*laplace_w[offset] - imdtdz[offset] - imdxdz[offset]*du[offset] - imdydz[offset]*dv[offset] - beta*psi_w[offset]*w[offset];
			float coeff3 = coeff + imdzdz[offset] + beta*psi_w[offset];
			dw[offset] = (1-omega)*dw[offset] + omega/coeff3*sigma3;
		}
	}
	
	/******************************* for ADMM  ********************************************/

	__global__
	void proximalF_RedBlack_Kernel(float* du, float* dv, float* dw, const float* imdxdx, const float* imdxdy, const float* imdxdz, const float* imdydy, const float* imdydz, const float* imdzdz, 
					const float* imdtdx, const float* imdtdy, const float* imdtdz, const float* laplace_u, const float* laplace_v, const float*laplace_w, 
					const float* u, const float* z_u, const float* v, const float* z_v,const float* w, const float* z_w,
					const int width, const int height, const int depth, const float alpha, const float beta, const float lambda, const float omega, const bool redKernel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		int rest = (x+y)%2;

		int start = redKernel ? rest : (1-rest);
		
		for(int z = start;z < depth;z += 2)
		{
			int offset = z*height*width+y*width+x;
			float sigma1 = 0, sigma2 = 0, sigma3 = 0, coeff = 0;
			float _weight;

			if(x > 0)
			{
				_weight = 1;
				sigma1 += _weight*du[offset-1];
				sigma2 += _weight*dv[offset-1];
				sigma3 += _weight*dw[offset-1];
				coeff  += _weight;
			}
			if(x < width-1)
			{
				_weight = 1;
				sigma1 += _weight*du[offset+1];
				sigma2 += _weight*dv[offset+1];
				sigma3 += _weight*dw[offset+1];
				coeff  += _weight;
			}
			if(y > 0)
			{
				_weight = 1;
				sigma1 += _weight*du[offset-width];
				sigma2 += _weight*dv[offset-width];
				sigma3 += _weight*dw[offset-width];
				coeff  += _weight;
			}
			if(y < height-1)
			{
				_weight = 1;
				sigma1 += _weight*du[offset+width];
				sigma2 += _weight*dv[offset+width];
				sigma3 += _weight*dw[offset+width];
				coeff  += _weight;
			}
			if(z > 0)
			{
				_weight = 1;
				sigma1 += _weight*du[offset-height*width];
				sigma2 += _weight*dv[offset-height*width];
				sigma3 += _weight*dw[offset-height*width];
				coeff  += _weight;
			}
			if(z < depth-1)
			{
				_weight = 1;
				sigma1 += _weight*du[offset+height*width];
				sigma2 += _weight*dv[offset+height*width];
				sigma3 += _weight*dw[offset+height*width];
				coeff  += _weight;
			}
			sigma1 *= alpha;
			sigma2 *= alpha;
			sigma3 *= alpha;
			coeff  *= alpha;
			// compute u
			sigma1 += alpha*laplace_u[offset] - imdtdx[offset] - imdxdy[offset]*dv[offset] - imdxdz[offset]*dw[offset] - beta*u[offset] - 0.5f*lambda*(u[offset] - z_u[offset]);
			float coeff1 = coeff + imdxdx[offset] + beta + 0.5f*lambda;
			du[offset] = (1-omega)*du[offset] + omega/coeff1*sigma1;
			// compute v
			sigma2 += alpha*laplace_v[offset] - imdtdy[offset] - imdxdy[offset]*du[offset] - imdydz[offset]*dw[offset] - beta*v[offset] - 0.5f*lambda*(v[offset] - z_v[offset]);
			float coeff2 = coeff + imdydy[offset] + beta + 0.5f*lambda;
			dv[offset] = (1-omega)*dv[offset] + omega/coeff2*sigma2;
			// compute w
			sigma3 += alpha*laplace_w[offset] - imdtdz[offset] - imdxdz[offset]*du[offset] - imdydz[offset]*dv[offset] - beta*w[offset] - 0.5f*lambda*(w[offset] - z_w[offset]);
			float coeff3 = coeff + imdzdz[offset] + beta + 0.5f*lambda;
			dw[offset] = (1-omega)*dw[offset] + omega/coeff3*sigma3;
		}
	}
	
	__global__
	void proximal_F2_Kernel(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* warpU, const float* warpV, const float* warpW,
							const int width, const int height, const int depth, const float gama, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			
			u[offset] = (gama*warpU[offset]+0.5*lambda*z_u[offset])/(gama+0.5*lambda);
			v[offset] = (gama*warpV[offset]+0.5*lambda*z_v[offset])/(gama+0.5*lambda);
			w[offset] = (gama*warpW[offset]+0.5*lambda*z_w[offset])/(gama+0.5*lambda);
		}
	}
	
	__global__
	void compute_z_u_z_v_z_w_for_proximal_F1_Kernel(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
								const float* u_for_q1, const float* v_for_q1, const float* w_for_q1, const int width, const int height, const int depth, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			z_u[offset] = u_for_F1[offset] - 1.0/lambda*u_for_q1[offset];
			z_v[offset] = v_for_F1[offset] - 1.0/lambda*v_for_q1[offset];
			z_w[offset] = w_for_F1[offset] - 1.0/lambda*w_for_q1[offset];
		}
	}
	
	__global__
	void compute_z_u_z_v_z_w_for_proximal_F2_Kernel(float* z_u, float* z_v, float* z_w, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
								const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			z_u[offset] = u_for_F2[offset] - 1.0/lambda*u_for_q2[offset];
			z_v[offset] = v_for_F2[offset] - 1.0/lambda*v_for_q2[offset];
			z_w[offset] = w_for_F2[offset] - 1.0/lambda*w_for_q2[offset];
		}
	}
	
	__global__
	void compute_z_u_z_v_z_w_for_proximal_G_Kernel(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
									const float* u_for_F2, const float* v_for_F2, const float* w_for_F2, const float* u_for_q1, const float* v_for_q1, const float* w_for_q1,
									const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			z_u[offset] = 0.5*(u_for_F1[offset] + 1.0/lambda*u_for_q1[offset] + u_for_F2[offset] + 1.0/lambda*u_for_q2[offset]);
			z_v[offset] = 0.5*(v_for_F1[offset] + 1.0/lambda*v_for_q1[offset] + v_for_F2[offset] + 1.0/lambda*v_for_q2[offset]);
			z_w[offset] = 0.5*(w_for_F1[offset] + 1.0/lambda*w_for_q1[offset] + w_for_F2[offset] + 1.0/lambda*w_for_q2[offset]);
		}
	}
	
	__global__
	void update_u_v_w_for_q1_q2_Kernel(float* u_for_q1, float* v_for_q1, float* w_for_q1, float* u_for_q2, float* v_for_q2, float* w_for_q2,
									 const float* u_for_F1, const float* v_for_F1, const float* w_for_F1, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
									 const float* u_for_G, const float* v_for_G, const float* w_for_G, const int width, const int height, const int depth, const float lambda)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if(x >= width || y >= height)
			return ;

		for(int z = 0;z < depth;z++)
		{
			int offset = z*height*width+y*width+x;
			u_for_q1[offset] += lambda*(u_for_F1[offset] - u_for_G[offset]);
			v_for_q1[offset] += lambda*(v_for_F1[offset] - v_for_G[offset]);
			w_for_q1[offset] += lambda*(w_for_F1[offset] - w_for_G[offset]);
			u_for_q2[offset] += lambda*(u_for_F2[offset] - u_for_G[offset]);
			v_for_q2[offset] += lambda*(v_for_F2[offset] - v_for_G[offset]);
			w_for_q2[offset] += lambda*(w_for_F2[offset] - w_for_G[offset]);
		}
	}
	
	/****************************************************************************************************/
	
	void cu_Compute_z_u_z_v_z_w_for_proximal_F1(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
							const float* u_for_q1, const float* v_for_q1, const float* w_for_q1, const int width, const int height, const int depth, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		compute_z_u_z_v_z_w_for_proximal_F1_Kernel<<<gridSize,blockSize>>>(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_q1,v_for_q1,w_for_q1,width,height,depth,lambda);
	}
	
	void cu_Compute_z_u_z_v_z_w_for_proximal_F2(float* z_u, float* z_v, float* z_w, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
							const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		compute_z_u_z_v_z_w_for_proximal_F2_Kernel<<<gridSize,blockSize>>>(z_u,z_v,z_w,u_for_F2,v_for_F2,w_for_F2,u_for_q2,v_for_q2,w_for_q2,width,height,depth,lambda);
	}
	
	void cu_Compute_z_u_z_v_z_w_for_proximal_G(float* z_u, float* z_v, float* z_w, const float* u_for_F1, const float* v_for_F1, const float* w_for_F1,
							const float* u_for_F2, const float* v_for_F2, const float* w_for_F2, const float* u_for_q1, const float* v_for_q1, const float* w_for_q1,
							const float* u_for_q2, const float* v_for_q2, const float* w_for_q2, const int width, const int height, const int depth, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		compute_z_u_z_v_z_w_for_proximal_G_Kernel<<<gridSize,blockSize>>>(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,
			u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,width,height,depth,lambda);
	}
	
	void cu_Update_u_v_w_for_q1_q2(float* u_for_q1, float* v_for_q1, float* w_for_q1, float* u_for_q2, float* v_for_q2, float* w_for_q2,
							const float* u_for_F1, const float* v_for_F1, const float* w_for_F1, const float* u_for_F2, const float* v_for_F2, const float* w_for_F2,
							const float* u_for_G, const float* v_for_G, const float* w_for_G, const int width, const int height, const int depth, const float lambda)
	{
		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		update_u_v_w_for_q1_q2_Kernel<<<gridSize,blockSize>>>(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
			u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_G,v_for_G,w_for_G,width,height,depth,lambda);
	}
	
	void cu_GetDerivatives(float* imdx, float* imdy, float* imdz, float* imdt, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels)
	{

		float* tmpBuf = 0;
		float* im1 = 0;
		float* im2 = 0;
		checkCudaErrors( cudaMalloc((void**)&tmpBuf, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&im1, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&im2, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(tmpBuf,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(im1,0,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(im2,0,sizeof(float)*width*height*depth*nChannels) );


		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		ZQ_CUDA_ImageProcessing3D::Imfilter_h_Gaussian_Kernel<<<gridSize,blockSize>>>(im1,Im1,width,height,depth,nChannels);
		ZQ_CUDA_ImageProcessing3D::Imfilter_v_Gaussian_Kernel<<<gridSize,blockSize>>>(tmpBuf,im1,width,height,depth,nChannels);
		ZQ_CUDA_ImageProcessing3D::Imfilter_d_Gaussian_Kernel<<<gridSize,blockSize>>>(im1,tmpBuf,width,height,depth,nChannels);

		ZQ_CUDA_ImageProcessing3D::Imfilter_h_Gaussian_Kernel<<<gridSize,blockSize>>>(im2,Im2,width,height,depth,nChannels);
		ZQ_CUDA_ImageProcessing3D::Imfilter_v_Gaussian_Kernel<<<gridSize,blockSize>>>(tmpBuf,im2,width,height,depth,nChannels);
		ZQ_CUDA_ImageProcessing3D::Imfilter_d_Gaussian_Kernel<<<gridSize,blockSize>>>(im2,tmpBuf,width,height,depth,nChannels);

		/* tmpBuf = im1*0.4 + im2*0.6 */
		ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmpBuf,im1,0.4,im2,0.6,width,height,depth,nChannels);

		/* imdx = \partial_x {tmpBuf} */
		ZQ_CUDA_ImageProcessing3D::Derivative_x_Advanced_Kernel<<<gridSize,blockSize>>>(imdx,tmpBuf,width,height,depth,nChannels);

		/* imdy = \partial_y {tmpBuf} */
		ZQ_CUDA_ImageProcessing3D::Derivative_y_Advanced_Kernel<<<gridSize,blockSize>>>(imdy,tmpBuf,width,height,depth,nChannels);
		
		/* imdz = \partial_z {tmpBuf} */
		ZQ_CUDA_ImageProcessing3D::Derivative_z_Advanced_Kernel<<<gridSize,blockSize>>>(imdz,tmpBuf,width,height,depth,nChannels);

		ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(imdt,im2,1,im1,-1,width,height,depth,nChannels);

		checkCudaErrors( cudaFree(tmpBuf) );
		checkCudaErrors( cudaFree(im1) );
		checkCudaErrors( cudaFree(im2) );
		tmpBuf = 0;
		im1 = 0;
		im2 = 0;
	}
	
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L2(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter)
	{
		float* du = 0;
		float* dv = 0;
		float* dw = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* laplace_w = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdz = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdxdz = 0;
		float* imdydy = 0;
		float* imdydz = 0;
		float* imdzdz = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* imdtdz = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_w, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdz, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdzdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdz, sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_w, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdz, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdzdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdz, 0, sizeof(float)*width*height*depth) );



		int nOuterFPIterations = nOuterFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v,w} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image tricubic */
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

			/* get imdx, imdy, imdt*/
			cu_GetDerivatives(imdx,imdy,imdz,imdt,Im1,warpIm2,width,height,depth,nChannels);

			/* reset du, dv, dw */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );

			/* compute imdxdx, imdxdy, imdxdz, imdydy, imdydz, imdzdz, imdtdx, imdtdy, imdtdz */
			compute_imdxdx_imdtdx_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,imdx,imdy,imdz,imdt,width,height,depth,nChannels);


			/* laplace u, v, w */
			ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_w,w,width,height,depth,1);


			// set omega
			float omega = 1.0f;
			float alpha2 = alpha*alpha;
			float beta2 = beta*beta;
		
			/* red - black solver begin */
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,u,v,w,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
					laplace_u,laplace_v,laplace_w,width,height,depth,alpha2,beta2,omega,true);
				OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,u,v,w,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
					laplace_u,laplace_v,laplace_w,width,height,depth,alpha2,beta2,omega,false);
			}
			/* red - black solver end */
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,1,width,height,depth,1);
		}
		/************       Outer Loop End     *************/

		/* warp image tricubic */
		ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

		
		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(dw) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(laplace_w) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdz) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdxdz) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdydz) );
		checkCudaErrors( cudaFree(imdzdz) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(imdtdz) );
		
		du = 0;
		dv = 0;
		dw = 0;
		laplace_u = 0;
		laplace_v = 0;
		laplace_w = 0;
		imdx = 0;
		imdy = 0;
		imdz = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdxdz = 0;
		imdydy = 0;
		imdydz = 0;
		imdzdz = 0;
		imdtdx = 0;
		imdtdy = 0;
		imdtdz = 0;
		
	}
	
	/*alpha: penality for velocity gradient*/
	void cu_OpticalFlow_L1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter,const int nSORIter)
	{
	
		float eps = optical_flow_L1_eps;
		float* du = 0;
		float* dv = 0;
		float* dw = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* laplace_w = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdz = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdxdz = 0;
		float* imdydy = 0;
		float* imdydz = 0;
		float* imdzdz = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* imdtdz = 0;
		float* psi_data = 0;
		float* psi_smooth = 0;
		float* psi_u = 0;
		float* psi_v = 0;
		float* psi_w = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_w, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdz, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdzdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_data,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_smooth,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_w, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdz, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdzdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_data,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_smooth,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_w,0,sizeof(float)*width*height*depth) );



		int nOuterFPIterations = nOuterFPIter;
		int nInnerFPIterations = nInnerFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{
			/* warp image tricubic */
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

			/* get imdx, imdy,imdz, imdt*/
			cu_GetDerivatives(imdx,imdy,imdz,imdt,Im1,warpIm2,width,height,depth,nChannels);

			/* reset du, dv, dw */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
			
			for(int inner_it = 0; inner_it < nInnerFPIterations;inner_it++)
			{
				/* compute psi_data*/
				compute_psi_data_Kernel<<<gridSize,blockSize>>>(psi_data,imdx,imdy,imdz,imdt,du,dv,dw,eps,width,height,depth,nChannels);

				/* compute imdxdx, imdxdy, imdxdz, imdydy, imdydz, imdzdz, imdtdx, imdtdy, imdtdz */
				compute_imdxdx_imdtdx_withpsidata_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
									imdx,imdy,imdz,imdt,psi_data,width,height,depth,nChannels);


				/*compute psi_smooth*/
				ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,1,width,height,depth,1);
				
				compute_psi_smooth_Kernel<<<gridSize,blockSize>>>(psi_smooth,u,v,w,eps,width,height,depth);
				
				compute_psi_u_v_w_Kernel<<<gridSize,blockSize>>>(psi_u,psi_v,psi_w,u,v,w,eps,width,height,depth);
				
				ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,-1,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,-1,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,-1,width,height,depth,1);

				/* laplace u, v, w with psi_smooth */
				Laplacian_withpsismooth_Kernel<<<gridSize,blockSize>>>(laplace_u,u,psi_smooth,width,height,depth);
				Laplacian_withpsismooth_Kernel<<<gridSize,blockSize>>>(laplace_v,v,psi_smooth,width,height,depth);
				Laplacian_withpsismooth_Kernel<<<gridSize,blockSize>>>(laplace_w,w,psi_smooth,width,height,depth);


				// set omega
				float omega = 1.0f;
				
			
				/* red - black solver begin */
				for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
				{
					OpticalFlow_L1_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,u,v,w,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
							laplace_u,laplace_v,laplace_w,psi_smooth,psi_u,psi_v,psi_w,width,height,depth,alpha,beta,omega,true);
					OpticalFlow_L1_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,u,v,w,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
							laplace_u,laplace_v,laplace_w,psi_smooth,psi_u,psi_v,psi_w,width,height,depth,alpha,beta,omega,false);
				}
				/* red - black solver end */
			}
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,1,width,height,depth,1);
		}
		/************       Outer Loop End     *************/

		/* warp image tricubic */
		ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);
		
		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(dw) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(laplace_w) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdz) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdxdz) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdydz) );
		checkCudaErrors( cudaFree(imdzdz) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(imdtdz) );
		checkCudaErrors( cudaFree(psi_data) );
		checkCudaErrors( cudaFree(psi_smooth) );
		checkCudaErrors( cudaFree(psi_u) );
		checkCudaErrors( cudaFree(psi_v) );
		checkCudaErrors( cudaFree(psi_w) );
		
		du = 0;
		dv = 0;
		dw = 0;
		laplace_u = 0;
		laplace_v = 0;
		laplace_w = 0;
		imdx = 0;
		imdy = 0;
		imdz = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdxdz = 0;
		imdydy = 0;
		imdydz = 0;
		imdzdz = 0;
		imdtdx = 0;
		imdtdy = 0;
		imdtdz = 0;
		psi_data = 0;
		psi_smooth = 0;
		psi_u = 0;
		psi_v = 0;
		psi_w = 0;
	}
	
	
	void cu_OpticalFlow_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter)
	{
		float eps = optical_flow_L1_eps;
		float* du = 0;
		float* dv = 0;
		float* dw = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* laplace_w = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdz = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdxdz = 0;
		float* imdydy = 0;
		float* imdydz = 0;
		float* imdzdz = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* imdtdz = 0;
		float* psi_data = 0;


		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_w, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdz, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdzdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_data,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_w, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdz, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdzdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_data,0,sizeof(float)*width*height*depth) );



		int nOuterFPIterations = nOuterFPIter;
		int nInnerFPIterations = nInnerFPIter;
		int nSORIterations = nSORIter;

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);


		/************       Outer Loop Begin     *************/
		//refresh {u,v,w} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image tricubic */
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

			/* get imdx, imdy, imdz, imdt*/
			cu_GetDerivatives(imdx,imdy,imdz,imdt,Im1,warpIm2,width,height,depth,nChannels);

			/* reset du, dv, dw */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
			
			for(int inner_it = 0; inner_it < nInnerFPIterations;inner_it++)
			{

				/* compute psi_data*/
				compute_psi_data_Kernel<<<gridSize,blockSize>>>(psi_data,imdx,imdy,imdz,imdt,du,dv,dw,eps,width,height,depth,nChannels);

				/* compute imdxdx, imdxdy, imdxdz, imdydy, imdydz, imdzdz, imdtdx, imdtdy, imdtdz */
				compute_imdxdx_imdtdx_withpsidata_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
								imdx,imdy,imdz,imdt,psi_data,width,height,depth,nChannels);


				/* laplace u, v, w with psi_smooth */
				ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_w,w,width,height,depth,1);


				// set omega
				float omega = 1.0;
				float alpha2 = alpha*alpha;
				float beta2 = beta*beta;
				
			
				/* red - black solver begin */
				for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
				{
					OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,u,v,w,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
							laplace_u,laplace_v,laplace_w,width,height,depth,alpha2,beta2,omega,true);
					OpticalFlow_L2_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,u,v,w,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
							laplace_u,laplace_v,laplace_w,width,height,depth,alpha2,beta2,omega,false);
				}
				/* red - black solver end */
			}
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,1,width,height,depth,1);
		}
		/************       Outer Loop End     *************/

		/* warp image tricubic */
		ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(dw) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(laplace_w) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdz) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdxdz) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdydz) );
		checkCudaErrors( cudaFree(imdzdz) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(imdtdz) );
		checkCudaErrors( cudaFree(psi_data) );
		
		du = 0;
		dv = 0;
		dw = 0;
		laplace_u = 0;
		laplace_v = 0;
		laplace_w = 0;
		imdx = 0;
		imdy = 0;
		imdz = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdxdz = 0;
		imdydy = 0;
		imdydz = 0;
		imdzdz = 0;
		imdtdx = 0;
		imdtdy = 0;
		imdtdz = 0;
		psi_data = 0;
	}
	
	void cu_Proximal_F1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* z_u, const float* z_v, const float* z_w,
				const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float lambda, const int nOuterFPIter, const int nSORIter)
	{
		float* du = 0;
		float* dv = 0;
		float* dw = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* laplace_w = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdz = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdxdz = 0;
		float* imdydy = 0;
		float* imdydz = 0;
		float* imdzdz = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* imdtdz = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_w, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdz, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdzdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdz, sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_w, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdz, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdzdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdz, 0, sizeof(float)*width*height*depth) );



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
		//refresh {u,v,w} in each loop
		for(int count = 0; count < nOuterFPIterations;count++)
		{

			/* warp image tricubic */
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

			/* get imdx, imdy, imdz,imdt*/
			cu_GetDerivatives(imdx,imdy,imdz,imdt,Im1,warpIm2,width,height,depth,nChannels);

			/* reset du, dv, dw */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );

			/* compute imdxdx, imdxdy, imdxdz, imdydy, imdydz, imdzdzm imdtdx, imdtdy, imdtdz */
			compute_imdxdx_imdtdx_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,imdx,imdy,imdz,imdt,width,height,depth,nChannels);


			/* laplace u, v, w */
			ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_w,w,width,height,depth,1);


			// set omega
			float omega = 1.0;
			float alpha2 = alpha*alpha;
			float beta2 = beta*beta;


			/* red - black solver begin */
			for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
			{
				proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
						laplace_u,laplace_v,laplace_w,u,z_u,v,z_v,w,z_w,width,height,depth,alpha2,beta2,lambda,omega,true);
				proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
						laplace_u,laplace_v,laplace_w,u,z_u,v,z_v,w,z_w,width,height,depth,alpha2,beta2,lambda,omega,false);
			}
			/* red - black solver end */
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,1,width,height,depth,1);
		}
		/************       Outer Loop End     *************/

		/* warp image bicubic */
		ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(dw) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(laplace_w) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdz) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdxdz) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdydz) );
		checkCudaErrors( cudaFree(imdzdz) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(imdtdz) );
		
		du = 0;
		dv = 0;
		dw = 0;
		laplace_u = 0;
		laplace_v = 0;
		laplace_w = 0;
		imdx = 0;
		imdy = 0;
		imdz = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdxdz = 0;
		imdydy = 0;
		imdydz = 0;
		imdzdz = 0;
		imdtdx = 0;
		imdtdy = 0;
		imdtdz = 0;
	}
	
	void cu_Proximal_F1_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* z_u, const float* z_v, const float* z_w,
					const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float lambda, 
					const int nOuterFPIter, const int nInnerFPIter, const int nSORIter)
	{
		
		float eps = optical_flow_L1_eps;
		float* du = 0;
		float* dv = 0;
		float* dw = 0;
		float* laplace_u = 0;
		float* laplace_v = 0;
		float* laplace_w = 0;
		float* imdx = 0;
		float* imdy = 0;
		float* imdz = 0;
		float* imdt = 0;
		float* imdxdx = 0;
		float* imdxdy = 0;
		float* imdxdz = 0;
		float* imdydy = 0;
		float* imdydz = 0;
		float* imdzdz = 0;
		float* imdtdx = 0;
		float* imdtdy = 0;
		float* imdtdz = 0;
		float* psi_data = 0;

		checkCudaErrors( cudaMalloc((void**)&du, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dv, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&dw, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_u, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_v, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&laplace_w, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdx, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdy, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdz, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdt, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&imdxdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdxdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdydz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdzdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdx, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdy, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&imdtdz, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&psi_data, sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_u, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_v, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(laplace_w, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdx, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdy, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdz, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdt, 0, sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemset(imdxdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdxdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdydz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdzdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdx, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdy, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(imdtdz, 0, sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(psi_data,0, sizeof(float)*width*height*depth) );



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

			/* warp image tricubic */
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

			/* get imdx, imdy, imdz, imdt*/
			cu_GetDerivatives(imdx,imdy,imdz,imdt,Im1,warpIm2,width,height,depth,nChannels);

			/* reset du, dv, dw */
			checkCudaErrors( cudaMemset(du, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dv, 0, sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(dw, 0, sizeof(float)*width*height*depth) );
			
			for(int inner_it = 0; inner_it < nInnerFPIterations; inner_it++)
			{
				/*compute psi_data*/
				compute_psi_data_Kernel<<<gridSize,blockSize>>>(psi_data,imdx,imdy,imdz,imdt,du,dv,dw,eps,width,height,depth,nChannels);

				/* compute imdxdx, imdxdy, imdxdz, imdydy, imdydz, imdzdz, imdtdx, imdtdy, imdtdz */
				compute_imdxdx_imdtdx_withpsidata_Kernel<<<gridSize,blockSize>>>(imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,imdx,imdy,imdz,imdt,psi_data,
																									width,height,depth,nChannels);


				/* laplace u, v, w */
				ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_u,u,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_v,v,width,height,depth,1);
				ZQ_CUDA_ImageProcessing3D::Laplacian_Kernel<<<gridSize,blockSize>>>(laplace_w,w,width,height,depth,1);


				// set omega
				float omega = 1.0;
				float alpha2 = alpha*alpha;
				float beta2 = beta*beta;


				/* red - black solver begin */
				for(int sor_it = 0;sor_it < nSORIterations;sor_it++)
				{
					proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
							laplace_u,laplace_v,laplace_w,u,z_u,v,z_v,w,z_w,width,height,depth,alpha2,beta2,lambda,omega,true);
					proximalF_RedBlack_Kernel<<<gridSize,blockSize>>>(du,dv,dw,imdxdx,imdxdy,imdxdz,imdydy,imdydz,imdzdz,imdtdx,imdtdy,imdtdz,
							laplace_u,laplace_v,laplace_w,u,z_u,v,z_v,w,z_w,width,height,depth,alpha2,beta2,lambda,omega,false);
				}
				/* red - black solver end */
			}
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u,du,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v,dv,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w,dw,1,width,height,depth,1);
		}
		/************       Outer Loop End     *************/

		/* warp image tricubic */
		ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpIm2,Im1,Im2,u,v,w,width,height,depth,nChannels);

		checkCudaErrors( cudaFree(du) ); 
		checkCudaErrors( cudaFree(dv) );
		checkCudaErrors( cudaFree(dw) );
		checkCudaErrors( cudaFree(laplace_u) );
		checkCudaErrors( cudaFree(laplace_v) );
		checkCudaErrors( cudaFree(laplace_w) );
		checkCudaErrors( cudaFree(imdx) );
		checkCudaErrors( cudaFree(imdy) );
		checkCudaErrors( cudaFree(imdz) );
		checkCudaErrors( cudaFree(imdt) );
		checkCudaErrors( cudaFree(imdxdx) );
		checkCudaErrors( cudaFree(imdxdy) );
		checkCudaErrors( cudaFree(imdxdz) );
		checkCudaErrors( cudaFree(imdydy) );
		checkCudaErrors( cudaFree(imdydz) );
		checkCudaErrors( cudaFree(imdzdz) );
		checkCudaErrors( cudaFree(imdtdx) );
		checkCudaErrors( cudaFree(imdtdy) );
		checkCudaErrors( cudaFree(imdtdz) );
		checkCudaErrors( cudaFree(psi_data) );
		
		du = 0;
		dv = 0;
		dw = 0;
		laplace_u = 0;
		laplace_v = 0;
		laplace_w = 0;
		imdx = 0;
		imdy = 0;
		imdz = 0;
		imdt = 0;
		imdxdx = 0;
		imdxdy = 0;
		imdxdz = 0;
		imdydy = 0;
		imdydz = 0;
		imdzdz = 0;
		imdtdx = 0;
		imdtdy = 0;
		imdtdz = 0;
		psi_data = 0;
	}
	
	void cu_Proximal_F2_first(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* next_u, const float* next_v, const float* next_w,
				const int width, const int height, const int depth, const float gama, const float lambda, const int nFPIter, const int nPoissonIter)
	{
		int nOuterFPIterations = nFPIter;
		int nPoissonIterations = nPoissonIter;

		float* warpU = 0;
		float* warpV = 0;
		float* warpW = 0;

		checkCudaErrors( cudaMalloc((void**)&warpU,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpV,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpW,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpU,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpV,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpW,0,sizeof(float)*width*height*depth) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		for(int out_it = 0;out_it < nOuterFPIterations;out_it++)
		{
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpU,u,next_u,u,v,w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpV,v,next_v,u,v,w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpW,w,next_w,u,v,w,width,height,depth,1);

			ZQ_CUDA_PoissonSolver3D::cu_SolveOpenPoissonRedBlack_Regular(warpU,warpV,warpW,width,height,depth,nPoissonIterations);

			proximal_F2_Kernel<<<gridSize,blockSize>>>(u,v,w,z_u,z_v,z_w,warpU,warpV,warpW,width,height,depth,gama,lambda);
		}

		checkCudaErrors( cudaFree(warpU) );
		checkCudaErrors( cudaFree(warpV) );
		checkCudaErrors( cudaFree(warpW) );
		warpU = 0;
		warpV = 0;
		warpW = 0;
	}
	
	void cu_Proximal_F2_middle(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* pre_u, const float* pre_v, const float* pre_w,
						const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, 
						const float gama, const float lambda, const int nFPIter, const int nPoissonIter)
	{
		int nOuterFPIterations = nFPIter;
		int nPoissonIterations = nPoissonIter;

		float* warpU_pre = 0;
		float* warpV_pre = 0;
		float* warpW_pre = 0;
		float* warpU_nex = 0;
		float* warpV_nex = 0;
		float* warpW_nex = 0;
		float* tmp_u = 0;
		float* tmp_v = 0;
		float* tmp_w = 0;

		checkCudaErrors( cudaMalloc((void**)&warpU_pre,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpV_pre,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpW_pre,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpU_nex,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpV_nex,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpW_nex,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_w,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpU_pre,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpV_pre,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpW_pre,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpU_nex,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpV_nex,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpW_nex,0,sizeof(float)*width*height*depth) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		for(int out_it = 0;out_it < nOuterFPIterations;out_it++)
		{
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpU_nex,u,next_u,u,v,w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpV_nex,v,next_v,u,v,w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpW_nex,w,next_w,u,v,w,width,height,depth,1);

			checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(tmp_w,0,sizeof(float)*width*height*depth) );
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_w,w,-1,width,height,depth,1);

			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpU_pre,u,pre_u,tmp_u,tmp_v,tmp_w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpV_pre,v,pre_v,tmp_u,tmp_v,tmp_w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpW_pre,w,pre_w,tmp_u,tmp_v,tmp_w,width,height,depth,1);

			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmp_u,warpU_pre,0.5,warpU_nex,0.5,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmp_v,warpV_pre,0.5,warpV_nex,0.5,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(tmp_w,warpW_pre,0.5,warpW_nex,0.5,width,height,depth,1);


			ZQ_CUDA_PoissonSolver3D::cu_SolveOpenPoissonRedBlack_Regular(tmp_u,tmp_v,tmp_w,width,height,depth,nPoissonIterations);

			proximal_F2_Kernel<<<gridSize,blockSize>>>(u,v,w,z_u,z_v,z_w,tmp_u,tmp_v,tmp_w,width,height,depth,2*gama,lambda);
		}

		checkCudaErrors( cudaFree(warpU_pre) );
		checkCudaErrors( cudaFree(warpV_pre) );
		checkCudaErrors( cudaFree(warpW_pre) );
		checkCudaErrors( cudaFree(warpU_nex) );
		checkCudaErrors( cudaFree(warpV_nex) );
		checkCudaErrors( cudaFree(warpW_nex) );
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		checkCudaErrors( cudaFree(tmp_w) );
		warpU_pre = 0;
		warpV_pre = 0;
		warpW_pre = 0;
		warpU_nex = 0;
		warpV_nex = 0;
		warpW_nex = 0;
		tmp_u = 0;
		tmp_v = 0;
		tmp_w = 0;
	}
	
	void cu_Proximal_F2_last(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const float* pre_u, const float* pre_v, const float* pre_w,
							const int width, const int height, const int depth, const float gama, const float lambda, const int nFPIter, const int nPoissonIter)
	{
		int nOuterFPIterations = nFPIter;
		int nPoissonIterations = nPoissonIter;

		float* warpU = 0;
		float* warpV = 0;
		float* warpW = 0;
		float* tmp_u = 0;
		float* tmp_v = 0;
		float* tmp_w = 0;

		checkCudaErrors( cudaMalloc((void**)&warpU,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpV,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&warpW,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&tmp_w,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpU,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpV,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(warpW,0,sizeof(float)*width*height*depth) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

		for(int out_it = 0;out_it < nOuterFPIterations;out_it++)
		{
			checkCudaErrors( cudaMemset(tmp_u,0,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(tmp_v,0,sizeof(float)*width*height*depth) );
			checkCudaErrors( cudaMemset(tmp_w,0,sizeof(float)*width*height*depth) );
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_u,u,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_v,v,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(tmp_w,w,-1,width,height,depth,1);

			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpU,u,pre_u,tmp_u,tmp_v,tmp_w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpV,v,pre_v,tmp_u,tmp_v,tmp_w,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::WarpImage_Tricubic_Kernel<<<gridSize,blockSize>>>(warpW,w,pre_w,tmp_u,tmp_v,tmp_w,width,height,depth,1);

			ZQ_CUDA_PoissonSolver3D::cu_SolveOpenPoissonRedBlack_Regular(warpU,warpV,warpW,width,height,depth,nPoissonIterations);

			proximal_F2_Kernel<<<gridSize,blockSize>>>(u,v,w,z_u,z_v,z_w,warpU,warpV,warpW,width,height,depth,gama,lambda);
		}

		checkCudaErrors( cudaFree(warpU) );
		checkCudaErrors( cudaFree(warpV) );
		checkCudaErrors( cudaFree(warpW) );
		checkCudaErrors( cudaFree(tmp_u) );
		checkCudaErrors( cudaFree(tmp_v) );
		checkCudaErrors( cudaFree(tmp_w) );
		warpU = 0;
		warpV = 0;
		warpW = 0;
		tmp_u = 0;
		tmp_v = 0;
		tmp_w = 0;
	}
	
	void cu_Proximal_G(float* u, float* v, float* w, const float* z_u, const float* z_v, const float* z_w, const int width, const int height, const int depth, const int nPoissonIter)
	{
		checkCudaErrors( cudaMemcpy(u,z_u,sizeof(float)*width*height*depth, cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v,z_v,sizeof(float)*width*height*depth, cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w,z_w,sizeof(float)*width*height*depth, cudaMemcpyDeviceToDevice) );

		int nPoissonIterations = nPoissonIter;

		ZQ_CUDA_PoissonSolver3D::cu_SolveOpenPoissonRedBlack_Regular(u,v,w,width,height,depth,nPoissonIterations);
	}
	
	void cu_OpticalFlow_ADMM(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter)
	{
		float* u_for_F = u;
		float* v_for_F = v;
		float* w_for_F = w;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q = 0;
		float* v_for_q = 0;
		float* w_for_q = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_G,1,u_for_q,-1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_G,1,v_for_q,-1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_w,w_for_G,1,w_for_q,-1.0,width,height,depth,1);

			
			cu_Proximal_F1(u_for_F,v_for_F,w_for_F,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_F,1,u_for_q,1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_F,1,v_for_q,1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_w,w_for_F,1,w_for_q,1.0,width,height,depth,1);

			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_F,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_G,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_F,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_G,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w_for_q,w_for_F,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w_for_q,w_for_G,-1,width,height,depth,1);
		}
		
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q) );
		checkCudaErrors( cudaFree(v_for_q) );
		checkCudaErrors( cudaFree(w_for_q) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		
		 u_for_F = 0;
		 v_for_F = 0;
		 w_for_F = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 w_for_G = 0;
		 u_for_q = 0;
		 v_for_q = 0;
		 w_for_q = 0;
		 z_u = 0;
		 z_v = 0;
		 z_w = 0;
	}
	
	void cu_OpticalFlow_ADMM_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels, 
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, 
						const int nSORIter, const int nPoissonIter)
	{
		float* u_for_F = u;
		float* v_for_F = v;
		float* w_for_F = w;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q = 0;
		float* v_for_q = 0;
		float* w_for_q = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );

		dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
		dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
		
		for(int it = 0;it < ADMMIter;it++)
		{
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_G,1,u_for_q,-1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_G,1,v_for_q,-1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_w,w_for_G,1,w_for_q,-1.0,width,height,depth,1);

			
			cu_Proximal_F1_DL1(u_for_F,v_for_F,w_for_F,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_u,u_for_F,1,u_for_q,1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_v,v_for_F,1,v_for_q,1.0,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Add_Im1_weight1_Im2_weight2_Kernel<<<gridSize,blockSize>>>(z_w,w_for_F,1,w_for_q,1.0,width,height,depth,1);

			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);
			
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_F,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(u_for_q,u_for_G,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_F,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(v_for_q,v_for_G,-1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w_for_q,w_for_F,1,width,height,depth,1);
			ZQ_CUDA_ImageProcessing3D::Addwith_Kernel<<<gridSize,blockSize>>>(w_for_q,w_for_G,-1,width,height,depth,1);

		}
		
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q) );
		checkCudaErrors( cudaFree(v_for_q) );
		checkCudaErrors( cudaFree(w_for_q) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		
		 u_for_F = 0;
		 v_for_F = 0;
		 w_for_F = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 w_for_G = 0;
		 u_for_q = 0;
		 v_for_q = 0;
		 w_for_q = 0;
		 z_u = 0;
		 z_v = 0;
		 z_w = 0;
	}
	
	void cu_OpticalFlow_ADMM_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* w_for_F1 = w;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* w_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* w_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* w_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_F2,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );


		float new_gamma = gamma*alpha*alpha;

		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_z_w_for_proximal_F1(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q1,v_for_q1,w_for_q1,width,height,depth,1);

			cu_Proximal_F1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_F2(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q2,v_for_q2,w_for_q2,width,height,depth,1);

			cu_Proximal_F2_first(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,next_u,next_v,next_w,width,height,depth,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_G(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
											width,height,depth,1);
		
			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);
			
			cu_Update_u_v_w_for_q1_q2(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,
											u_for_G,v_for_G,w_for_G,width,height,depth,1);
		}
		
		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(w_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(w_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(w_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		

		 u_for_F1 = 0;
		 v_for_F1 = 0;
		 w_for_F1 = 0;
		 u_for_F2 = 0;
		 v_for_F2 = 0;
		 w_for_F2 = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 w_for_G = 0;
		 u_for_q1 = 0;
		 v_for_q1 = 0;
		 w_for_q1 = 0;
		 u_for_q2 = 0;
		 v_for_q2 = 0;
		 w_for_q2 = 0;
		 z_u = 0;
		 z_v = 0;
		 z_w = 0;
	}
	
	void cu_OpticalFlow_ADMM_DL1_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* w_for_F1 = w;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* w_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* w_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* w_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_F2,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );


		float new_gamma = gamma*alpha*alpha;

		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_z_w_for_proximal_F1(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q1,v_for_q1,w_for_q1,width,height,depth,1);

			cu_Proximal_F1_DL1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_F2(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q2,v_for_q2,w_for_q2,width,height,depth,1);

			cu_Proximal_F2_first(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,next_u,next_v,next_w,width,height,depth,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_G(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
											width,height,depth,1);
		
			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);
			
			cu_Update_u_v_w_for_q1_q2(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,
											u_for_G,v_for_G,w_for_G,width,height,depth,1);
		}
		
		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(w_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(w_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(w_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		

		 u_for_F1 = 0;
		 v_for_F1 = 0;
		 w_for_F1 = 0;
		 u_for_F2 = 0;
		 v_for_F2 = 0;
		 w_for_F2 = 0;
		 u_for_G = 0;
		 v_for_G = 0;
		 w_for_G = 0;
		 u_for_q1 = 0;
		 v_for_q1 = 0;
		 w_for_q1 = 0;
		 u_for_q2 = 0;
		 v_for_q2 = 0;
		 w_for_q2 = 0;
		 z_u = 0;
		 z_v = 0;
		 z_w = 0;
	}
	
	void cu_OpticalFlow_ADMM_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							   const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* w_for_F1 = w;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* w_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* w_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* w_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_F2,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );


		float new_gamma = gamma*alpha*alpha;


		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_z_w_for_proximal_F1(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q1,v_for_q1,w_for_q1,width,height,depth,1);
			
			cu_Proximal_F1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_F2(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q2,v_for_q2,w_for_q2,width,height,depth,1);

			cu_Proximal_F2_middle(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,pre_u,pre_v,pre_w,next_u,next_v,next_w,width,height,depth,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_G(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
									width,height,depth,1);

			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);

			cu_Update_u_v_w_for_q1_q2(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_G,v_for_G,w_for_G,
									width,height,depth,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(w_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(w_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(w_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		

		u_for_F1 = 0;
		v_for_F1 = 0;
		w_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		w_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		w_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		w_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		w_for_q2 = 0;
		z_u = 0;
		z_v = 0;
		z_w = 0;
	}
	
	void cu_OpticalFlow_ADMM_DL1_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							   const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
							   const float alpha, const float beta, const float gamma, const float lambda, 
							   const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* w_for_F1 = w;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* w_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* w_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* w_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_F2,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );


		float new_gamma = gamma*alpha*alpha;


		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_z_w_for_proximal_F1(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q1,v_for_q1,w_for_q1,width,height,depth,1);
			
			cu_Proximal_F1_DL1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_F2(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q2,v_for_q2,w_for_q2,width,height,depth,1);

			cu_Proximal_F2_middle(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,pre_u,pre_v,pre_w,next_u,next_v,next_w,width,height,depth,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_G(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
									width,height,depth,1);

			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);

			cu_Update_u_v_w_for_q1_q2(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_G,v_for_G,w_for_G,
									width,height,depth,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(w_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(w_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(w_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		

		u_for_F1 = 0;
		v_for_F1 = 0;
		w_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		w_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		w_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		w_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		w_for_q2 = 0;
		z_u = 0;
		z_v = 0;
		z_w = 0;
	}
	
	
	void cu_OpticalFlow_ADMM_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* w_for_F1 = w;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* w_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* w_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* w_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_F2,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );

		float new_gamma = gamma*alpha*alpha;
		
		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_z_w_for_proximal_F1(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q1,v_for_q1,w_for_q1,width,height,depth,1);

			cu_Proximal_F1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nSORIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_F2(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q2,v_for_q2,w_for_q2,width,height,depth,1);

			cu_Proximal_F2_last(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,pre_u,pre_v,pre_w,width,height,depth,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_G(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
							width,height,depth,1);

			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);

			cu_Update_u_v_w_for_q1_q2(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,
							u_for_G,v_for_G,w_for_G,width,height,depth,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(w_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(w_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(w_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		
		u_for_F1 = 0;
		v_for_F1 = 0;
		w_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		w_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		w_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		w_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		w_for_q2 = 0;
		z_u = 0;
		z_v = 0;
		z_w = 0;	
	}

	void cu_OpticalFlow_ADMM_DL1_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
							  const int ADMMIter, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter, const int nWarpFPIter, const int nPoissonIter)
	{
		float* u_for_F1 = u;
		float* v_for_F1 = v;
		float* w_for_F1 = w;
		float* u_for_F2 = 0;
		float* v_for_F2 = 0;
		float* w_for_F2 = 0;
		float* u_for_G = 0;
		float* v_for_G = 0;
		float* w_for_G = 0;
		float* u_for_q1 = 0;
		float* v_for_q1 = 0;
		float* w_for_q1 = 0;
		float* u_for_q2 = 0;
		float* v_for_q2 = 0;
		float* w_for_q2 = 0;
		float* z_u = 0;
		float* z_v = 0;
		float* z_w = 0;

		checkCudaErrors( cudaMalloc((void**)&u_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_F2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_G,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q1,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_for_q2,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_u,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_v,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&z_w,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(u_for_F2,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_F2,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_F2,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(u_for_G,u_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(v_for_G,v_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(w_for_G,w_for_F1,sizeof(float)*width*height*depth,cudaMemcpyDeviceToDevice) );

		checkCudaErrors( cudaMemset(u_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q1,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(u_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(v_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(w_for_q2,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_u,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_v,0,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMemset(z_w,0,sizeof(float)*width*height*depth) );

		float new_gamma = gamma*alpha*alpha;
		
		for(int it = 0;it < ADMMIter;it++)
		{
			cu_Compute_z_u_z_v_z_w_for_proximal_F1(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q1,v_for_q1,w_for_q1,width,height,depth,1);

			cu_Proximal_F1_DL1(u_for_F1,v_for_F1,w_for_F1,warpIm2,Im1,Im2,z_u,z_v,z_w,width,height,depth,nChannels,alpha,beta,lambda,nOuterFPIter,nInnerFPIter,nSORIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_F2(z_u,z_v,z_w,u_for_G,v_for_G,w_for_G,u_for_q2,v_for_q2,w_for_q2,width,height,depth,1);

			cu_Proximal_F2_last(u_for_F2,v_for_F2,w_for_F2,z_u,z_v,z_w,pre_u,pre_v,pre_w,width,height,depth,new_gamma,lambda,nWarpFPIter,nPoissonIter);

			cu_Compute_z_u_z_v_z_w_for_proximal_G(z_u,z_v,z_w,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,
							width,height,depth,1);

			cu_Proximal_G(u_for_G,v_for_G,w_for_G,z_u,z_v,z_w,width,height,depth,nPoissonIter);

			cu_Update_u_v_w_for_q1_q2(u_for_q1,v_for_q1,w_for_q1,u_for_q2,v_for_q2,w_for_q2,u_for_F1,v_for_F1,w_for_F1,u_for_F2,v_for_F2,w_for_F2,
							u_for_G,v_for_G,w_for_G,width,height,depth,1);
		}

		checkCudaErrors( cudaFree(u_for_F2) );
		checkCudaErrors( cudaFree(v_for_F2) );
		checkCudaErrors( cudaFree(w_for_F2) );
		checkCudaErrors( cudaFree(u_for_G) );
		checkCudaErrors( cudaFree(v_for_G) );
		checkCudaErrors( cudaFree(w_for_G) );
		checkCudaErrors( cudaFree(u_for_q1) );
		checkCudaErrors( cudaFree(v_for_q1) );
		checkCudaErrors( cudaFree(w_for_q1) );
		checkCudaErrors( cudaFree(u_for_q2) );
		checkCudaErrors( cudaFree(v_for_q2) );
		checkCudaErrors( cudaFree(w_for_q2) );
		checkCudaErrors( cudaFree(z_u) );
		checkCudaErrors( cudaFree(z_v) );
		checkCudaErrors( cudaFree(z_w) );
		
		u_for_F1 = 0;
		v_for_F1 = 0;
		w_for_F1 = 0;
		u_for_F2 = 0;
		v_for_F2 = 0;
		w_for_F2 = 0;
		u_for_G = 0;
		v_for_G = 0;
		w_for_G = 0;
		u_for_q1 = 0;
		v_for_q1 = 0;
		w_for_q1 = 0;
		u_for_q2 = 0;
		v_for_q2 = 0;
		w_for_q2 = 0;
		z_u = 0;
		z_v = 0;
		z_w = 0;	
	}
	
	
	/***********************************************************************/
	
	extern "C"
	void InitDevice3D(const int deviceid)
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
	float OpticalFlow3D_L2(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_L2(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,width,height,depth,nChannels,alpha,beta,nOuterFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
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
	float OpticalFlow3D_L1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter,const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_L1(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,width,height,depth,nChannels,alpha,beta,nOuterFPIter,nInnerFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		
		u_d = 0;
		v_d = 0;
		w_d = 0;
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
	float OpticalFlow3D_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
						const float alpha, const float beta, const int nOuterFPIter, const int nInnerFPIter, const int nSORIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) ); 
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels,cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_DL1(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,width,height,depth,nChannels,alpha,beta,nOuterFPIter,nInnerFPIter,nSORIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth,cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels,cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		
		u_d = 0;
		v_d = 0;
		w_d = 0;
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
	float OpticalFlow3D_ADMM(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
						const float alpha, const float beta, const float lambda, const int ADMMIter, const int nOuterFPIter, const int nSORIter, const int nPoissonIter)
	{
		float time = 0;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );

		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,width,height,depth,nChannels,alpha,beta,lambda,ADMMIter,nOuterFPIter,nSORIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		u_d = 0;
		v_d = 0;
		w_d = 0;
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
	float OpticalFlow3D_ADMM_DL1(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const int width, const int height, const int depth, const int nChannels,
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
		float* w_d = 0;
		float* Im1_d = 0;
		float* Im2_d = 0;
		float* warpIm2_d = 0;

		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );

		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,width,height,depth,nChannels,alpha,beta,lambda,ADMMIter,nOuterFPIter,nInnerFPIter,nSORIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		u_d = 0;
		v_d = 0;
		w_d = 0;
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
	float OpticalFlow3D_ADMM_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
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
		float* next_w_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_w_d,next_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_First(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,next_u_d,next_v_d,next_w_d,width,height,depth,nChannels,alpha,beta,gamma,lambda,
									ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(next_w_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		u_d = 0;
		v_d = 0;
		w_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		next_w_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	extern "C"
	float OpticalFlow3D_ADMM_DL1_First(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* next_u, const float* next_v, const float* next_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
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
		float* next_w_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_w_d,next_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1_First(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,next_u_d,next_v_d,next_w_d,width,height,depth,nChannels,alpha,beta,gamma,lambda,
							ADMMIter,nOuterFPIter,nInnerFPIter, nSORIter,nWarpFPIter,nPoissonIter);

		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(next_w_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		
		u_d = 0;
		v_d = 0;
		w_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		next_w_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time; 
	}
	
	extern "C"
	float OpticalFlow3D_ADMM_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							   const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
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
		float* next_w_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* pre_w_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_w_d,next_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_w_d,pre_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Middle(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,pre_w_d,next_u_d,next_v_d,next_w_d,width,height,depth,nChannels,
									alpha,beta,gamma,lambda,ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);
	
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(next_w_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(pre_w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		next_w_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		pre_w_d = 0;
		u_d = 0;
		v_d = 0;
		w_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	
	extern "C"
	float OpticalFlow3D_ADMM_DL1_Middle(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							   const float* next_u, const float* next_v, const float* next_w, const int width, const int height, const int depth, const int nChannels, 
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
		float* next_w_d = 0;
		float* pre_u_d = 0;
		float* pre_v_d = 0;
		float* pre_w_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;

		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&next_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&next_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_u_d,next_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_v_d,next_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(next_w_d,next_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_w_d,pre_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1_Middle(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,pre_w_d,next_u_d,next_v_d,next_w_d,width,height,depth,nChannels,
									alpha,beta,gamma,lambda,ADMMIter,nOuterFPIter,nInnerFPIter,nSORIter,nWarpFPIter,nPoissonIter);
	
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(next_u_d) );
		checkCudaErrors( cudaFree(next_v_d) );
		checkCudaErrors( cudaFree(next_w_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(pre_w_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		next_u_d = 0;
		next_v_d = 0;
		next_w_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		pre_w_d = 0;
		u_d = 0;
		v_d = 0;
		w_d = 0;
		
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;
	}
	
	extern "C"
	float OpticalFlow3D_ADMM_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
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
		float* pre_w_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		
		
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_w_d,pre_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_Last(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,pre_w_d,width,height,depth,nChannels,alpha,beta,gamma,lambda,
							ADMMIter,nOuterFPIter,nSORIter,nWarpFPIter,nPoissonIter);
		
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(pre_w_d) );
		
		u_d = 0;
		v_d = 0;
		w_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		pre_w_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
	extern "C"
	float OpticalFlow3D_ADMM_DL1_Last(float* u, float* v, float* w, float* warpIm2, const float* Im1, const float* Im2, const float* pre_u, const float* pre_v, const float* pre_w,
							  const int width, const int height, const int depth, const int nChannels, const float alpha, const float beta, const float gamma, const float lambda, 
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
		float* pre_w_d = 0;
		float* u_d = 0;
		float* v_d = 0;
		float* w_d = 0;
		
		
		checkCudaErrors( cudaMalloc((void**)&Im1_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&Im2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&warpIm2_d,sizeof(float)*width*height*depth*nChannels) );
		checkCudaErrors( cudaMalloc((void**)&pre_u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&pre_w_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&u_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&v_d,sizeof(float)*width*height*depth) );
		checkCudaErrors( cudaMalloc((void**)&w_d,sizeof(float)*width*height*depth) );

		checkCudaErrors( cudaMemcpy(Im1_d,Im1,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(Im2_d,Im2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(warpIm2_d,warpIm2,sizeof(float)*width*height*depth*nChannels, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_u_d,pre_u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_v_d,pre_v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(pre_w_d,pre_w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(u_d,u,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(v_d,v,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(w_d,w,sizeof(float)*width*height*depth, cudaMemcpyHostToDevice) );
		
		cu_OpticalFlow_ADMM_DL1_Last(u_d,v_d,w_d,warpIm2_d,Im1_d,Im2_d,pre_u_d,pre_v_d,pre_w_d,width,height,depth,nChannels,alpha,beta,gamma,lambda,
							ADMMIter,nOuterFPIter,nInnerFPIter,nSORIter,nWarpFPIter,nPoissonIter);
		
		checkCudaErrors( cudaMemcpy(u,u_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(v,v_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(w,w_d,sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(warpIm2,warpIm2_d,sizeof(float)*width*height*depth*nChannels, cudaMemcpyDeviceToHost) );
		
		checkCudaErrors( cudaFree(u_d) );
		checkCudaErrors( cudaFree(v_d) );
		checkCudaErrors( cudaFree(w_d) );
		checkCudaErrors( cudaFree(warpIm2_d) );
		checkCudaErrors( cudaFree(Im1_d) );
		checkCudaErrors( cudaFree(Im2_d) );
		checkCudaErrors( cudaFree(pre_u_d) );
		checkCudaErrors( cudaFree(pre_v_d) );
		checkCudaErrors( cudaFree(pre_w_d) );
		
		u_d = 0;
		v_d = 0;
		w_d = 0;
		Im1_d = 0;
		Im2_d = 0;
		warpIm2_d = 0;
		pre_u_d = 0;
		pre_v_d = 0;
		pre_w_d = 0;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		return time;	
	}
	
}

#endif