#include "cudaLib.cuh"
#include "assert.h"
#include <chrono>
using namespace std::chrono;

#define DEBUG_PRINT_DISABLE
//#define MEDIAN_NOPT


void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}


int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	ImageDim imgDim;
	uint8_t *imgData, *imgData_dev;
	uint8_t *outData;

	// Sanitize Inputs
	if(args.filterH > 8 || args.filterW > 8) {
		std::cout << "Maximum supported filter size is 8 x 8.\n";
		return -1;
	}
	if(args.filterH != args.filterW) {
		std::cout << "Filer must be square.\n";
		return -1;
	}
	
	int bytesRead = loadBytesImage(imgPath, imgDim, &imgData);
	int imgSize = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;
	if(cudaMalloc(&imgData_dev, imgSize * sizeof(uint8_t)) != cudaSuccess || cudaMallocManaged(&outData, imgSize * sizeof(uint8_t)) != cudaSuccess) {
		printf("Unable to cudaMalloc memory ... Exiting!\n");
		return -1;
	}
	std::cout << "Size = " << imgSize << "\n";

	// Copy image to device
	if(cudaMemcpy(imgData_dev, imgData, imgSize, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Failed to cudaMemcpy image to device... Exiting!\n");
		return -1;
	}

	// Divide image ~ 1 thread per channel
	float blockWidth = 96;
	float blockHeight = 2;
	dim3 bDims(blockWidth, blockHeight, imgDim.channels);
	dim3 gDims(ceil(imgDim.width / blockWidth), ceil(imgDim.height / blockHeight));
	uint32_t tileSize = abs(blockWidth + args.filterW - 1) * abs(blockHeight + args.filterH - 1) * imgDim.channels;

	auto start = high_resolution_clock::now();
	medianFilter_gpu<<<gDims, bDims, tileSize * sizeof(uint8_t)>>>(imgData_dev, imgDim, outData, args);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	auto stop = high_resolution_clock::now();

	// Output
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Execution time was: " << duration.count() << " ms\n";
	writeBytesImage(outPath, imgDim, outData);
	return 0;
}


__global__ 
void medianFilter_gpu (uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {

	// Thread variables
	uint32_t tx = threadIdx.x + blockDim.x * blockIdx.x; // Thread x-index in grid/image
	uint32_t ty = threadIdx.y + blockDim.y * blockIdx.y; // Thread y-index in grid/image
	uint32_t blkX = blockDim.x * blockIdx.x; // x-index of threadblock in grid/image
	uint32_t blkY = blockDim.y * blockIdx.y; // y-index of threadblock in grid/image
	uint32_t tidxBlk = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; // 1D index of the thread in the block (row-major)
	uint32_t blkSize = blockDim.x * blockDim.y * blockDim.z; // Number of threads in the block

	// Filter/Tile variables
	int filterOff = (args.filterW - 1) / 2; // Offset of the pixel block within the tile (from top left ~ square window)
	uint32_t tileWidth = blockDim.x + args.filterW - 1; // Tile width in pixels
	uint32_t tileHeight = blockDim.y + args.filterH - 1; // Tile height in pixels
	uint32_t windowSize = args.filterW * args.filterH;

	// Shared Memory
	extern __shared__ uint8_t localData[]; // Shared memory block
	uint8_t* localTile = localData; // Shared tile for the thread block
	uint8_t window[64]; // Maximum window size is 8x8
	assert(windowSize <= 64);


	// Have the threads load the tile
	#ifdef MEDIAN_NOPT
	int loadNum = ceilf((float) (tileWidth * tileHeight) / blkSize); // Number of pixels to load for each thread
	uint32_t loadOff = tidxBlk * loadNum;

	for(int tilePixel = loadOff; tilePixel < loadOff + loadNum; tilePixel++) {
		uint32_t tileX = tilePixel % tileWidth;
		uint32_t tileY = tilePixel / tileWidth;
		int32_t inX = (tileX + blkX - filterOff);
		int32_t inY = (tileY + blkY - filterOff);
		int32_t inputPixel = inX + inY * imgDim.width;

		if(tilePixel < tileWidth * tileHeight) {
			for(int channel = 0; channel < imgDim.channels; channel++) {
				if(inX >= 0 && inY >=0 && inputPixel < imgDim.width * imgDim.height) {
					localTile[channel + imgDim.channels * (tileX + tileY * tileWidth)] = inPixels[inputPixel * imgDim.channels + channel];	
				}
				else {
					localTile[channel + imgDim.channels * (tileX + tileY * tileWidth)] = 128;
				}
			}
		}
	}
	#else
	int loadNum = ceilf((float) (tileWidth * tileHeight) / blkSize) * imgDim.channels; // Number of bytes/channels each thread loads

	for(uint32_t element = tidxBlk; element < tidxBlk + blkSize * loadNum; element += blkSize) {

		if(element < tileWidth * tileHeight * imgDim.channels) {
			uint32_t pixelNum = element / imgDim.channels;
			int inX = pixelNum % tileWidth + blkX - filterOff;
			int inY = pixelNum / tileWidth + blkY - filterOff;
			int inputPixel = inX + inY * imgDim.width;

			if(inX >= 0 && inY >= 0 && inputPixel < imgDim.width * imgDim.height) {
				localTile[element] = inPixels[inputPixel * imgDim.channels + (element % imgDim.channels)];	
			}
			else {
				localTile[element] = 128;
			}
		}
	}
	#endif
	__syncthreads();

	// Perform median filter
	if(tx < imgDim.width && ty < imgDim.height) {

		// Iterate through each channel
		//for(int channel = 0; channel < imgDim.channels; channel++) {
			uint32_t windowIdx = 0;
			uint32_t channel = threadIdx.z;
			
			for(uint32_t windowX = threadIdx.x; windowX < args.filterW + threadIdx.x; windowX++) {
				for(uint32_t windowY = threadIdx.y; windowY < args.filterH + threadIdx.y; windowY++){
					window[windowIdx++] = localTile[channel + imgDim.channels * (windowX + windowY * tileWidth)];
				}
			}

			// Sort the window
			dim3 arr(windowSize, 0, 0);
			sortPixels_gpu(window, arr);

			// Set the output pixel to the median
			uint8_t median = windowSize % 2 ? window[windowSize / 2] : (window[windowSize / 2] + window[(windowSize / 2) - 1]) / 2;
			outPixels[channel + imgDim.channels * (tx + (ty * imgDim.width))] = median;
		//}
	}
}


__device__ // Do not call with more than 1 thread per array
void sortPixels_gpu (uint8_t * array, dim3 arrayDim) {

	// Selection Sort
	for(int i = 0; i < arrayDim.x - 1; i++) {
		int min = i;

		for(int j = i + 1; j < arrayDim.x; j++){
			min = array[j] < array[min] ? j : min;
		}

		if(min != i) {
			uint8_t temp = array[min];
			array[min] = array[i];
			array[i] = temp;
		}
	}
}


int runGpuPool (TensorShape inShape, PoolLayerArgs poolArgs) {
	srand(time(NULL));

	// Sanitize Inputs
	if(inShape.width < poolArgs.poolW || inShape.height < poolArgs.poolH) {
		std::cout << "Pool size cannot exceed input size.\n";
		return -1;
	}
	if(inShape.width < 1 || inShape.height < 1 || inShape.channels < 1) {
		std::cout << "Invalid input matrix.\n";
		return -1;
	}
	if(poolArgs.poolW < 1 || poolArgs.poolH < 1 || poolArgs.strideW < 1 || poolArgs.strideH < 1) {
		std::cout << "Invalid pool arguements.\n";
		return -1;
	}

	// Create the input matrix (NCHW)
	float* inMatrix;
	uint32_t inSize = inShape.width * inShape.height * inShape.channels;
	float inCPU[inShape.width * inShape.height * inShape.channels];

	
	if(cudaMallocManaged(&inMatrix, inSize * sizeof(float)) != cudaSuccess) {
		printf("Unable to cudaMalloc input memory ... Exiting!\n");
		return -1;
	}
	for(int i = 0; i < inSize; i++) {
		//inMatrix[i] = (float) i;
		inMatrix[i] = static_cast<float>(std::rand()) / static_cast<float>(std::rand());
	}
	memcpy(inCPU, inMatrix, inShape.height * inShape.width * inShape.channels * sizeof(float));

	// Allocate the output matrix
	int padding = 0; // No padding for now
	uint32_t outH = ((inShape.height - poolArgs.poolH + padding * 2) / poolArgs.strideH) + 1;
	uint32_t outW = ((inShape.width - poolArgs.poolW + padding * 2) / poolArgs.strideW) + 1;
	TensorShape outShape = {outH, outW, inShape.channels};
	uint32_t outSize = outH * outW * outShape.channels;
	float* outMatrix; 

	if(cudaMallocManaged(&outMatrix, outSize * sizeof(float)) != cudaSuccess) {
		printf("Unable to cudaMalloc output memory ... Exiting!\n");
		return -1;
	}

	// Split pooling per output pixel
	float blockSize = 1024;
	float blockDim = sqrt(blockSize);
	dim3 bDims(blockDim, blockDim);
	dim3 gDims(ceil(outShape.width / blockDim), ceil(outShape.height / blockDim));
	uint32_t tileH = ((outShape.height - 1) * poolArgs.strideH) - (padding * 2) + poolArgs.poolH;
	uint32_t tileW = ((outShape.width - 1) * poolArgs.strideW) - (padding * 2) + poolArgs.poolW;
	cudaStream_t streams[inShape.channels];

	// Run one stream/kernel per channel
	auto start = high_resolution_clock::now();
	for(uint32_t channel = 0; channel < inShape.channels; channel++) {
		float* input = &inMatrix[channel * inShape.width * inShape.height];
		float* output = &outMatrix[channel * outShape.width * outShape.height];

		cudaStreamCreate(&streams[channel]);
		poolLayer_gpu<<<gDims, bDims, tileH * tileW * sizeof(float), streams[channel]>>>(input, inShape, output, outShape, poolArgs);\
	}
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	auto stop = high_resolution_clock::now();

	// Run CPU version
	float outCPU[outShape.width * outShape.height * outShape.channels];
	poolLayer_cpu(inCPU, inShape, outCPU, outShape, poolArgs);
	// Count any errors 
	int errors = 0;
	for(int chan = 0; chan < inShape.channels; chan++) {
		for(int y = 0; y < outShape.height; y++) {
			for(int x = 0; x < outShape.width; x++) {
				uint32_t idx = chan * outShape.width * outShape.height + (x + y * outShape.width);
				if(outMatrix[idx] != outCPU[idx]) {
					errors++;
					#ifndef DEBUG_PRINT_DISABLE
					std::cout << "Error detected in x: " << x << " y: " << y << " channel: " << chan << "\n";
					std::cout << "Got: " << outMatrix[idx] << ", Expected: " << outCPU[idx] << ".\n\n";
					#endif
				}
			}
		}
	}
	std::cout << "Errors: " << errors << ".\n";

	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Execution time was: " << duration.count() << " ms\n";

	// Matrix printer
	#ifndef DEBUG_PRINT_DISABLE
	// Print input matrix
	for(int chan = 0; chan < inShape.channels; chan++) {
		std::cout << "\nINPUT MATRIX CHANNEL " << chan + 1 << "\n\n";
		for(int y = 0; y < inShape.height; y++) {
			for(int x = 0; x < inShape.width; x++) {
				std::cout << inMatrix[chan * inShape.width * inShape.height + (x + y * inShape.width)] << " ";
			}
			std::cout << "\n";
		}
	}
	std::cout << "\n";
	// Print output matrix
	for(int chan = 0; chan < inShape.channels; chan++) {
		std::cout << "\nOUTPUT MATRIX CHANNEL " << chan + 1 << "\n\n";
		for(int y = 0; y < outShape.height; y++) {
			for(int x = 0; x < outShape.width; x++) {
				std::cout << outMatrix[chan * outShape.width * outShape.height + (x + y * outShape.width)] << " ";
			}
			std::cout << "\n";
		}
	}
	#endif

	return errors;
}


__global__
void poolLayer_gpu (float * input, TensorShape inShape, float * output, TensorShape outShape, PoolLayerArgs args) {

	// Thread variables
	uint32_t tx = threadIdx.x + blockDim.x * blockIdx.x; // Thread x-index in grid
	uint32_t ty = threadIdx.y + blockDim.y * blockIdx.y; // Thread y-index in grid
	uint32_t blkX = blockDim.x * blockIdx.x; // x-index of first thread in threadblock in grid
	uint32_t blkY = blockDim.y * blockIdx.y; // y-index of first thread in threadblock in grid
	uint32_t tidxBlk = threadIdx.x + threadIdx.y * blockDim.x; // 1D index of the thread in the block (row-major)

	// Pool/Tile variables
	uint32_t padding = 0; // Padding for the matrix (unused)
	uint32_t tileHeight = ((outShape.height - 1) * args.strideH) - (padding * 2) + args.poolH; // Height of shared tile
	uint32_t tileWidth = ((outShape.width - 1) * args.strideW) - (padding * 2) + args.poolW; // Width of shared tile
	extern __shared__ float localTile[]; // Shared tile for the thread block
	
	// Have the threads load the tile
	uint32_t blkSize = blockDim.x * blockDim.y; // Number of threads in the block
	int loadNum = ceilf((float) (tileWidth * tileHeight) / blkSize); // Number of elements each thread loads

	for(uint32_t element = tidxBlk; element < tidxBlk + blkSize * loadNum; element += blkSize) {
		if(element < tileWidth * tileHeight) {
			uint32_t inX = element % tileWidth + blkX * args.strideW;
			uint32_t inY = element / tileWidth + blkY * args.strideH;
			uint32_t inputElement = inX + inY * inShape.width;

			if(inputElement < inShape.width * inShape.height) {
				localTile[element] = input[inputElement];	
			}
		}
	}
	__syncthreads();

	// Perform pooling
	if(tx < outShape.width && ty < outShape.height) {
			uint32_t tileX = threadIdx.x * args.strideW;
			uint32_t tileY = threadIdx.y * args.strideH;
			float poolPick;

			for(uint32_t poolY = tileY; poolY < args.poolH + tileY; poolY++) {
				for(uint32_t poolX = tileX; poolX < args.poolW + tileX; poolX++){
					float element = localTile[poolX + poolY * tileWidth];
					
					if(poolX == tileX && poolY == tileY){
						poolPick = args.opType == PoolOp::AvgPool ? 0 : element;
					}
					else {
						switch (args.opType)
						{
							case PoolOp::MaxPool:
								poolPick = element > poolPick ? element : poolPick;
								break;
							case PoolOp::MinPool:
								poolPick = element < poolPick ? element : poolPick;
								break;
							case PoolOp::AvgPool:
								poolPick += element;
								break;
						}
					}
				}
			}

			poolPick = args.opType == PoolOp::AvgPool ? poolPick / (args.poolH * args.poolW) : poolPick;
			output[tx + ty * outShape.width] = poolPick;
	}
}
