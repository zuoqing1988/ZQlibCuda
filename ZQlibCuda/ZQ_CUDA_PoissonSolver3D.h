#ifndef _ZQ_CUDA_POISSON_SOLVER_3D_H_
#define _ZQ_CUDA_POISSON_SOLVER_3D_H_

namespace ZQ_CUDA_PoissonSolver3D
{
	/*********************  Extern "C" functions   *************************/

	/*******         Open Poisson    ********/
	/*First Implementation*/
	extern "C" 
	void SolveOpenPoissonRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int maxIter);

	extern "C"
	void SolveOpenPoissonRedBlack3D_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int maxIter);

	extern "C" 
	void SolveOpenPoissonRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth, const int maxIter);

	extern "C" 
	void SolveOpenPoissonRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const int width, const int height, const int depth, const int maxIter);

	/******         Closed Poisson   *******/
	extern "C"
	void SolveClosedPoissonRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float div_per_volume, const int width, const int height, const int depth, const int maxIter);

	extern "C"
	void SolveClosedPoissonRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float div_per_volume, const int width, const int height, const int depth, const int maxIter);

	extern "C"
	void SolveClosedPoissonRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
										const float div_per_volume, const int width, const int height, const int depth, const int maxIter);

	/********           OpenFlux     **********/
	/*First Implementation*/
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" void SolveOpenFluxRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const int width, const int height, const int depth, const int outerIter, const int innerIter);

	extern "C" void SolveOpenFluxRedBlack3D_Regular(float* u, float* v, float* w, const int width, const int height, const int depth, const int outerIter, const int innerIter);

	extern "C" void SolveOpenFluxRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const int width, const int height, const int depth, const int outerIter, const int innerIter);

	extern "C" void SolveOpenFluxRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW,
					const int width, const int height, const int depth, const int outerIter, const int innerIter);

	/********           ClosedFlux     **********/
	/*First Implementation*/
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" void SolveClosedFluxRedBlack3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float div_per_volume, const int width, const int height, const int depth, const int outerIter, const int innerIter);

	extern "C" void SolveClosedFluxRedBlackwithOccupy3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* occupy, const float div_per_volume, const int width, const int height, const int depth, const int outerIter, const int innerIter);

	extern "C" void SolveClosedFluxRedBlackwithFaceRatio3D_MAC(float* mac_u, float* mac_v, float* mac_w, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, const float* unoccupyW, const float div_per_volume,
					const int width, const int height, const int depth, const int outerIter, const int innerIter);


	/*** Open Octree Poisson ***/
	
	/*index: [x,y,z,num,offset]...
	* neighbor info : [level,x,y,z]...
	*/
	extern "C"
	float SolveOpenOctreePoissonRedBlack3_3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const int width, const int height, const int depth, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int level0_info_len_red, const int* level0_neighborinfo_red,
		const int level0_num_black, const int* level0_index_black, const int level0_info_len_black, const int* level0_neighborinfo_black,
		const int level1_num_red, const int* level1_index_red, const int level1_info_len_red, const int* level1_neighborinfo_red, 
		const int level1_num_black, const int* level1_index_black, const int level1_info_len_black, const int* level1_neighborinfo_black,
		const int level2_num_red, const int* level2_index_red, const int level2_info_len_red, const int* level2_neighborinfo_red,
		const int level2_num_black, const int* level2_index_black, const int level2_info_len_black, const int* level2_neighborinfo_black,
		const int level3_num_red, const int* level3_index_red, const int level3_info_len_red, const int* level3_neighborinfo_red,
		const int level3_num_black, const int* level3_index_black, const int level3_info_len_black, const int* level3_neighborinfo_black);

	/*** Closed Octree Poisson ***/

	/*index: [x,y,z,num,offset]...
	* neighbor info : [level,x,y,z]...
	*/
	extern "C"
	float SolveClosedOctreePoissonRedBlack3_3D_MAC(float* mac_u, float* mac_v, float* mac_w, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const float div_per_volume, const int width, const int height, const int depth, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int level0_info_len_red, const int* level0_neighborinfo_red,
		const int level0_num_black, const int* level0_index_black, const int level0_info_len_black, const int* level0_neighborinfo_black,
		const int level1_num_red, const int* level1_index_red, const int level1_info_len_red, const int* level1_neighborinfo_red, 
		const int level1_num_black, const int* level1_index_black, const int level1_info_len_black, const int* level1_neighborinfo_black,
		const int level2_num_red, const int* level2_index_red, const int level2_info_len_red, const int* level2_neighborinfo_red,
		const int level2_num_black, const int* level2_index_black, const int level2_info_len_black, const int* level2_neighborinfo_black,
		const int level3_num_red, const int* level3_index_red, const int level3_info_len_red, const int* level3_neighborinfo_red,
		const int level3_num_black, const int* level3_index_black, const int level3_info_len_black, const int* level3_neighborinfo_black);

}
#endif