/**
 * @file lab3.cpp
 * @author Abhishek Bhaumick (abhaumic@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2021-01-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include "lab3.cuh"
#include "cpuLib.h"
#include "cudaLib.cuh"

int main(int argc, char** argv) {
	std::string str;
	int choice;

	std::cout << "ECE 695 - Lab 2 \n";
	std::cout << "Select application: \n";
	std::cout << "  1 - CPU SAXPY \n";
	std::cout << "  2 - GPU SAXPY \n";
	std::cout << "  3 - CPU Monte-Carlo Pi \n";
	std::cout << "  4 - GPU Monte-Carlo Pi \n";
	std::cout << "  5 - Bytes-Image File Test \n";
	std::cout << "  6 - Median Filter CPU \n";
	std::cout << "  7 - Median Filter GPU \n";
	std::cout << "  100 - AlexNet on GPU \n";

	std::cin >> choice;

	std::cout << "\n";
	std::cout << "Choice selected - " << choice << "\n\n";

	MedianFilterArgs filArgs;
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

		case 5:
			std::cout << "Running BytesImage File Test on CPU! \n\n";
			testLoadBytesImage("./resources/lena512color.tiff.bytes");
			std::cout << "\n\n ... Done!\n";
			break;

		case 6:
			std::cout << "Running Median Filter on CPU! \n\n";
			filArgs = {4, 4};
			runCpuMedianFilter("./resources/lena512color.tiff.bytes", 
				"./resources/lena512color_fil.bytes", filArgs);
			std::cout << "\n\n ... Done!\n";
			break;

		case 7:
			std::cout << "Running Median Filter on GPU! \n\n";
			filArgs = {4, 4};
			runGpuMedianFilter("./resources/lena512color.tiff.bytes", 
				"./resources/lena512color_fil.bytes", filArgs);
			std::cout << "\n\n ... Done!\n";
			break;
		case 100:
		{
			std::cout << "Running AlexNet! \n\n";
			srand(time(NULL));
			bool printen = false;

			// Initial Input 
			uint32_t batch = 5;
			TensorShape inShape = AlexL1_InShape;
			inShape.count = batch;
			PoolLayerArgs pArgs = {PoolOp::MaxPool, 3, 3, 2, 2}; // Max pool, 3x3, strides of 2
			float* inTensor = NULL;
			inTensor = cudaMakeTensorUnified(inShape);
			TensorObject in = {inShape, inTensor};
			TensorObject out;

			// Conv
			auto start = std::chrono::high_resolution_clock::now();
			runGpuConv(&out, &in, AlexL1_FilterShape, AlexL1_ConvArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Max Pool
			runGpuPool(&out, &in, pArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Conv
			runGpuConv(&out, &in, AlexL2_FilterShape, AlexL2_ConvArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Max Pool
			runGpuPool(&out, &in, pArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Conv
			runGpuConv(&out, &in, AlexL3_FilterShape, AlexL3_ConvArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Conv
			runGpuConv(&out, &in, AlexL4_FilterShape, AlexL4_ConvArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }			

			// Conv
			runGpuConv(&out, &in, AlexL5_FilterShape, AlexL5_ConvArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Max Pool
			runGpuPool(&out, &in, pArgs);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// Flatten the Layer (batch, 1, 1, 9216)
			in.shape.channels = 1;
			in.shape.height = 1;
			in.shape.width = out.shape.width * out.shape.height * out.shape.channels;

			// GEMM
			TensorShape weights = {batch, 1, in.shape.width, 4096};
			runGpuGemm(&out, &in, weights);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// GEMM
			weights = {batch, 1, 4096, 4096};
			runGpuGemm(&out, &in, weights);
			in.shape = out.shape;
			in.tensor = out.tensor;
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			// GEMM
			weights = {batch, 1, 4096, 1000};
			runGpuGemm(&out, &in, weights);
			if(printen) { std::cout << out.shape << " Ptr: " << out.tensor << "\n"; }

			auto stop = std::chrono::high_resolution_clock::now();
			auto durationGPU = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << "AlexNet final output size is: " << out.shape << "\n";
			std::cout << "AlexNet execution time was: " << durationGPU.count() << " ms\n";
			break;
		}
		default:
			std::cout << "Hmm ... Devious, you are!\n";
			std::cout << " Choose correctly, you must.\n";
			break;
	}

	return 0;
}

int testLoadBytesImage(std::string filePath) {
	ImageDim imgDim;
	uint8_t * imgData;
	int bytesRead = loadBytesImage(filePath, imgDim, &imgData);
	int bytesExpected = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;
	if (bytesRead != bytesExpected) {
		std::cout << "Read Failed - Insufficient Bytes - " << bytesRead 
			<< " / "  << bytesExpected << " \n";
		return -1;
	}
	std::cout << "Read Success - " << bytesRead << " Bytes \n"; 
	return 0;
}


