/**
 * @file lab1.cpp
 * @author Abhishek Bhaumick (abhaumic@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2021-01-11
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include "lab1.cuh"
#include "cpuLib.h"
#include "cudaLib.cuh"

int main(int argc, char** argv) {
	std::string str;
	int choice;

	std::cout << "ECE 695 - Lab 1 \n";
	std::cout << "Select application: \n";
	std::cout << "  1 - CPU SAXPY \n";
	std::cout << "  2 - GPU SAXPY \n";
	std::cout << "  3 - CPU Monte-Carlo Pi \n";
	std::cout << "  4 - GPU Monte-Carlo Pi \n";

	std::cin >> choice;

	std::cout << "\n";
	std::cout << "Choice selected - " << choice << "\n\n";

	switch (choice) {
		//  CPU only SAXPY
		case 1:
			std::cout << "Running SAXPY CPU! \n\n";
			runCpuSaxpy(VECTOR_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		//  CUDA + GPU SAXPY
		case 2:
			std::cout << "Running SAXPY GPU! \n\n";
			runGpuSaxpy(VECTOR_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		case 3:
			std::cout << "Running Monte-Carlo Pi Estimation on CPU! \n\n";
			runCpuMCPi(MC_ITER_COUNT, MC_SAMPLE_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		case 4:
			std::cout << "Running Monte-Carlo Pi Estimation on GPU! \n\n";
			runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, REDUCE_BLOCKS, REDUCE_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		// GPU Profiling
		case 5:
			std::cout << "Secret Menu - Running GPU Profiling! \n\n";
			//runGpuSaxpy(pow(2,25));
			//runGpuMCPi(GENERATE_BLOCKS, pow(2,25), REDUCE_BLOCKS, pow(2,10));
			std::cout << "\n\n ... Done!\n";
			break;

		// GPU SAXPY Stress
		case 10:
			std::cout << "Secret Menu - Running SAXPY GPU Stress! \n\n";

			// "Special" cases
			for(int i = -2; i < 3; i++) {
				runGpuSaxpy(i);
			}

			// Range Test
			for(int i = 1; i <= 25; i++) {
				runGpuSaxpy(VECTOR_SIZE + i);
			}
			
			// Large Number Test
			for(int i = pow(2, 29) - 5; i < pow(2, 29); i++) {
				runGpuSaxpy(VECTOR_SIZE + i);
			}
			std::cout << "\n\n ... Done!\n";
			break;

		// GPU Monte-Carlo Pi Generate Stress
		case 11:
			std::cout << "Secret Menu - Running Monte-Carlo Pi Stress on GPU!  \n\n";

			// Range cases
			for(int i = 0; i < 100; i++) {
				runGpuMCPi(i, SAMPLE_SIZE - i, REDUCE_BLOCKS, REDUCE_SIZE);
			}
			for(int i = 0; i < 100; i++) {
				runGpuMCPi(GENERATE_BLOCKS + i, i, REDUCE_BLOCKS, REDUCE_SIZE);
			}
			
			// Large Number Test
				runGpuMCPi(GENERATE_BLOCKS, pow(2,29) - 1, REDUCE_BLOCKS, REDUCE_SIZE);
				runGpuMCPi(pow(2,29) - 1, GENERATE_BLOCKS, REDUCE_BLOCKS, REDUCE_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		// GPU Monte-Carlo Pi Reduce Stress
		case 12:
			std::cout << "Secret Menu - Running Monte-Carlo Pi Stress on GPU!  \n\n";

			// Range cases
			for(int i = 0; i < 100; i++) {
				runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, i, REDUCE_SIZE);
			}
			for(int i = 0; i < 100; i++) {
				runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, REDUCE_BLOCKS, i);
			}
			runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, 1024, GENERATE_BLOCKS + 1);
			runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, 1024, GENERATE_BLOCKS);
			runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, 1024, GENERATE_BLOCKS - 1);
			
			// Large Number Test
				runGpuMCPi(GENERATE_BLOCKS, pow(2,29) - 1, REDUCE_BLOCKS, REDUCE_SIZE);
				runGpuMCPi(pow(2,29) - 1, GENERATE_BLOCKS, REDUCE_BLOCKS, REDUCE_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		default:
			std::cout << "Hmm ... Devious, you are!\n";
			std::cout << " Choose correctly, you must.\n";
			break;
	}

	return 0;
}


