#ifndef _ZQ_CUDA_POISSON_SOLVER_2D_H_
#define _ZQ_CUDA_POISSON_SOLVER_2D_H_


namespace ZQ_CUDA_PoissonSolver2D
{
	/*********************  Extern "C" functions   *************************/

	/*******         Open Poisson    ********/
	extern "C" void SolveOpenPoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter);

	extern "C" void SolveOpenPoissonRedBlack2D_Regular(float* u, float* v, const int width, const int height, const int maxIter);

	extern "C" void SolveOpenPoissonRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter);

	extern "C" void SolveOpenPoissonRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV,
					const int width, const int height, const int maxIter);

	/*Another Implementation of Open Poisson*/
	extern "C" void SolveOpenPoissonRedBlack2_2D_MAC(float* mac_u, float* mac_v, const int width, const int height, const int maxIter);

	extern "C" void SolveOpenPoissonRedBlack2_2D_Regular(float* u, float* v, const int width, const int height, const int maxIter);

	extern "C" void SolveOpenPoissonRedBlackwithOccupy2_2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int maxIter);

	extern "C" void SolveOpenPoissonRedBlackwithFaceRatio2_2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const float* unoccupyU, const float* unoccupyV,
					const int width, const int height, const int maxIter);

	/**********      Closed Poisson   ************/
	extern "C" void SolveClosedPoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int maxIter);

	extern "C" void SolveClosedPoissonRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy,/* const int first_x, const int first_y,*/ const float div_per_volume,
					const int width, const int height, const int maxIter);

	extern "C" void SolveClosedPoissonRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV,
		/*const int first_x, const int first_y,*/ const float div_per_volume, const int width, const int height, const int maxIter);

	extern "C" void test_SolveClosedPoissonRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy,  /*const int first_x, const int first_y,*/ const float div_per_volume,
		const int width, const int height, const int maxIter);


	/********           OpenFlux     **********/
	/*First Implementation*/
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" void SolveOpenFluxRedBlack2D_MAC(float* mac_u, float* mac_v, const int width, const int height, const int outerIter, const int innerIter);

	extern "C" void SolveOpenFluxRedBlack2D_Regular(float* u, float* v, const int width, const int height, const int outerIter, const int innerIter);

	extern "C" void SolveOpenFluxRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const int width, const int height, const int outerIter, const int innerIter);

	extern "C" void SolveOpenFluxRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const float* unoccupyU, const float* unoccupyV,
					const int width, const int height, const int outerIter, const int innerIter);

	/********           ClosedFlux     **********/
	/*First Implementation*/
	/*outer iteration: Augmented Lagrange Multiplier method
	* inner iteration: red-black iteration
	*/
	extern "C" void SolveClosedFluxRedBlack2D_MAC(float* mac_u, float* mac_v, const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter);

	extern "C" void SolveClosedFluxRedBlackwithOccupy2D_MAC(float* mac_u, float* mac_v, const bool* occupy, const float div_per_volume, const int width, const int height, const int outerIter, const int innerIter);

	extern "C" void SolveClosedFluxRedBlackwithFaceRatio2D_MAC(float* mac_u, float* mac_v, const float* unoccupyVolume, const float* unoccupyU, const float* unoccupyV, const float div_per_volume,
					const int width, const int height, const int outerIter, const int innerIter);

	
	
	/*** Open Octree Poisson ***/
	extern "C"
	void SolveOpenOctreePoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const int width, const int height, const int maxIter);
	
	/*index : [x,y]...*/
	extern "C"
	void SolveOpenOctreePoissonRedBlack2_2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const int width, const int height, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int level0_num_black, const int* level0_index_black,
		const int level1_num_red, const int* level1_index_red, const int level1_num_black, const int* level1_index_black,
		const int level2_num_red, const int* level2_index_red, const int level2_num_black, const int* level2_index_black,
		const int level3_num_red, const int* level3_index_red, const int level3_num_black, const int* level3_index_black);

	/*index: [x,y,num,offset]...
	* neighbor info : [level,x,y]...
	*/
	extern "C"
	void SolveOpenOctreePoissonRedBlack3_2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const int width, const int height, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int level0_info_len_red, const int* level0_neighborinfo_red,
		const int level0_num_black, const int* level0_index_black, const int level0_info_len_black, const int* level0_neighborinfo_black,
		const int level1_num_red, const int* level1_index_red, const int level1_info_len_red, const int* level1_neighborinfo_red, 
		const int level1_num_black, const int* level1_index_black, const int level1_info_len_black, const int* level1_neighborinfo_black,
		const int level2_num_red, const int* level2_index_red, const int level2_info_len_red, const int* level2_neighborinfo_red,
		const int level2_num_black, const int* level2_index_black, const int level2_info_len_black, const int* level2_neighborinfo_black,
		const int level3_num_red, const int* level3_index_red, const int level3_info_len_red, const int* level3_neighborinfo_red,
		const int level3_num_black, const int* level3_index_black, const int level3_info_len_black, const int* level3_neighborinfo_black);

	/*** Closed Octree Poisson ***/
	extern "C"
	void SolveClosedOctreePoissonRedBlack2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const float div_per_volume, const int width, const int height, const int maxIter);
	
	/*index : [x,y]...*/
	extern "C"
	void SolveClosedOctreePoissonRedBlack2_2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const float div_per_volume, const int width, const int height, const int maxIter,
		const int level0_num_red, const int* level0_index_red, const int level0_num_black, const int* level0_index_black,
		const int level1_num_red, const int* level1_index_red, const int level1_num_black, const int* level1_index_black,
		const int level2_num_red, const int* level2_index_red, const int level2_num_black, const int* level2_index_black,
		const int level3_num_red, const int* level3_index_red, const int level3_num_black, const int* level3_index_black);

	/*index: [x,y,num,offset]...
	* neighbor info : [level,x,y]...
	*/
	extern "C"
	void SolveClosedOctreePoissonRedBlack3_2D_MAC(float* mac_u, float* mac_v, const bool* leaf0, const bool* leaf1, const bool* leaf2, const bool* leaf3,
		const float div_per_volume, const int width, const int height, const int maxIter,
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