
#include "cudaLib.cuh"
#include "assert.h"
#include <chrono>

// See individual sections for algrithm specifc defines & enables
#define DEBUG_PRINT_DISABLE // Disables debug printing
#define FP_ERROR 0.005 // Tolerated FP calculation error (actual value not %)
#define PARALLEL_INIT // Enables parallel tensor initialization

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


//*****************************************************************************
// SAXPY
//*****************************************************************************
__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < size) {
		x[i] = x[i] * scale + y[i];
	}
}
 

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";
	uint64_t vectorBytes = vectorSize * sizeof(float);
	float scale = 2.0f;

	if(vectorSize < 1) {
		printf("Please check your inputs.. Exiting!\n");
		return -1;
	}
	
	// Init the host arrays
	float* a;
	float* b;
	float* c;

	a = (float *) malloc(vectorBytes);
	b = (float *) malloc(vectorBytes);
	c = (float *) malloc(vectorBytes);

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!\n");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif
	
	// Alloc device arrays
	float* x_dev;
	float* y_dev;

	if(cudaMalloc(&x_dev, vectorBytes) != cudaSuccess || cudaMalloc(&y_dev, vectorBytes) != cudaSuccess) {
		printf("Unable to cudaMalloc memory ... Exiting!\n");
		return -1;
	}

	// Copy arrays from host to device
	if(cudaMemcpy(x_dev, a, vectorBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy for vector a failed... Exiting!\n");
		return -1;
	}
	if(cudaMemcpy(y_dev, b, vectorBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy for vector b failed... Exiting!\n");
		return -1;
	}
	
	// Run kernel
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(x_dev, y_dev, scale, vectorSize);
	
	// Sync for reporting
	cudaDeviceSynchronize();

	// Copy memory back from device
	if(cudaMemcpy(c, x_dev, vectorBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("Failed to cudaMemcpy result back to host... Exiting!\n");
		return -1;
	}

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	cudaFree(x_dev);
	cudaFree(y_dev);
	free(a);
	free(b);
	free(c);
	return 0;
}


//*****************************************************************************
// MONTE-CARLO PI
//*****************************************************************************
__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	curandState_t rng;
	curand_init(clock64() + i, 0, 0, &rng);
	

	// Generate and check
	if(i < pSumSize) {
		for(uint64_t j = 0; j < sampleSize; j++) {
			float x = curand_uniform(&rng);
    		float y = curand_uniform(&rng);

			if(static_cast<int>(x*x + y*y) == 0)
				pSums[i]++;
		}
	}
}


__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	if(reduceSize > pSumSize) { return; }
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	uint64_t totalThreads = blockDim.x * gridDim.x;

	// Each working thread will wrtie to a single (exclusive) output location
	if(i < reduceSize) {
		totals[i] = 0;

		// Calculate how many items this thread needs to sum
		uint64_t cnt = ceil(static_cast<float>(pSumSize) / min(reduceSize, totalThreads));
		uint64_t offsetInd = i * cnt;

		// Sum the partials
		if(offsetInd + cnt < pSumSize) {
			for(int j = 0; j < cnt; j++) {
				totals[i] += pSums[offsetInd + j];
			}
		}
		else if(pSumSize >= offsetInd) {
			for(int j = 0; j < (pSumSize - offsetInd); j++) {
				totals[i] += pSums[offsetInd + j];
			}
		}
	}
}


int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, uint64_t reduceThreadCount, uint64_t reduceSize) {

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
	std::cout << "It took " << time_span.count() << " seconds.\n";

	return 0;
}


double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, uint64_t reduceThreadCount, uint64_t reduceSize) {
	double approxPi = 0;
	uint64_t pSumBytes = generateThreadCount * sizeof(uint64_t);
	uint64_t rSumBytes = reduceSize * sizeof(uint64_t);
	
	// Input checking
	if(generateThreadCount < 1 || sampleSize < 1 || reduceThreadCount < 1 || reduceSize < 1 || (reduceSize > generateThreadCount)) {
		printf("Please check your inputs.. Exiting!\n");
		return -1;
	}

	// Allocate device memory
	uint64_t* pSum_dev;
	if(cudaMalloc(&pSum_dev, pSumBytes) != cudaSuccess) {
		printf("Unable to cudaMalloc memory ... Exiting!\n");
		return -1;
	}

	// Run generate kernel
	generatePoints<<<ceil(generateThreadCount / 256.0), 256>>>(pSum_dev, generateThreadCount, sampleSize);

	// Reduce the result
	uint64_t* rSum_dev;
	
	cudaDeviceSynchronize();
	if(cudaMalloc(&rSum_dev, rSumBytes) != cudaSuccess) {
		printf("Unable to cudaMalloc memory ... Exiting!\n");
		return -1;
	}

	cudaMemset(rSum_dev, 0, rSumBytes);
	reduceCounts<<<ceil(reduceThreadCount / 256.0), 256>>>(pSum_dev, rSum_dev, generateThreadCount, reduceSize);

	// Copy result back to deivce
	uint64_t* rSum;
	rSum = static_cast<uint64_t*>(malloc(rSumBytes));

	cudaDeviceSynchronize();
	if(cudaMemcpy(rSum, rSum_dev, rSumBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("Failed to cudaMemcpy result back to host... Exiting!\n");
		return -1;
	}

	// Estimate Pi
	for(int i = 0; i < reduceSize; i++) {
		approxPi += rSum[i];
	}

	approxPi = (approxPi / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;

	cudaFree(pSum_dev);
	cudaFree(rSum_dev);
	free(rSum);
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

	auto start = std::chrono::high_resolution_clock::now();
	medianFilter_gpu<<<gDims, bDims, tileSize * sizeof(uint8_t)>>>(imgData_dev, imgDim, outData, args);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	auto stop = std::chrono::high_resolution_clock::now();

	// Output
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Execution time was: " << duration.count() << " ms\n";
	writeBytesImage(outPath, imgDim, outData);
	return 0;
}


//*****************************************************************************
// MEDIAN FILTER
//*****************************************************************************
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


//*****************************************************************************
// POOLING
//*****************************************************************************
void runGpuPool (TensorObject* out, TensorObject* inObj, PoolLayerArgs poolArgs) {
	TensorShape outShape;
	TensorShape inShape = inObj->shape;

	// Sanitize Inputs
	assert(inShape.width >= poolArgs.poolW && inShape.height >= poolArgs.poolH);
	assert(inShape.width >= 1 && inShape.height >= 1 && inShape.channels >= 1 && inShape.count >= 1);
	assert(poolArgs.poolW >= 1 && poolArgs.poolH >= 1 && poolArgs.strideW >= 1 && poolArgs.strideH >= 1);

	// Get Input Tensor
	float* inMatrix = inObj->tensor;

	// Allocate the output matrix
	int padding = 0; // No padding for now
	outShape.count = inShape.count;
	outShape.channels = inShape.channels;
	outShape.height = ((inShape.height - poolArgs.poolH + padding * 2) / poolArgs.strideH) + 1;;
	outShape.width = ((inShape.width - poolArgs.poolW + padding * 2) / poolArgs.strideW) + 1;
	float *outMatrix; 
	assert(cudaMallocManaged(&outMatrix, tensorSize(outShape) * sizeof(float)) == cudaSuccess);

	// Split pooling per output pixel
	float blockSize = 1024;
	float blockDim = sqrt(blockSize);
	dim3 bDims(blockDim, blockDim);
	dim3 gDims(ceil(outShape.width / blockDim), ceil(outShape.height / blockDim));
	uint32_t tileH = ((outShape.height - 1) * poolArgs.strideH) - (padding * 2) + poolArgs.poolH;
	uint32_t tileW = ((outShape.width - 1) * poolArgs.strideW) - (padding * 2) + poolArgs.poolW;
	cudaStream_t streams[inShape.count * inShape.channels];

	// Run one stream/kernel per channel/kernel
	for(uint32_t count = 0; count < inShape.count; count++) {
		for(uint32_t channel = 0; channel < inShape.channels; channel++) {
			float* input = &inMatrix[count * inShape.channels * inShape.width * inShape.height + channel * inShape.width * inShape.height];
			float* output = &outMatrix[count * outShape.channels * outShape.width * outShape.height + channel * outShape.width * outShape.height];

			cudaStreamCreate(&streams[count * inShape.channels + channel]);
			poolLayer_gpu<<<gDims, bDims, tileH * tileW * sizeof(float), streams[count * inShape.channels + channel]>>>(input, inShape, output, outShape, poolArgs);
		}
	}
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	out->shape = outShape;
	out->tensor = outMatrix;
	cudaFree(inMatrix);
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


//*****************************************************************************
// CUDA TENSOR FUNCTIONS
//*****************************************************************************
float* cudaMakeTensor(TensorShape & shape) {
	float *host = NULL, *device;

	if(makeTensor(&host, shape) != 0) {
		std::cout << "Unable to make tensor.\n" ;
		return NULL;
	}
	if(cudaMalloc(&device, tensorSize(shape) * sizeof(float)) != cudaSuccess) {
		std::cout << "Unable to cudaMalloc tensor.\n" ;
		return NULL;
	}
	if(cudaMemcpy(device, host, tensorSize(shape) * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Failed to cudaMemcpy tensor.\n");
		return NULL;
	}
	cudaDeviceSynchronize();

	free(host);
	return device;
}


float* cudaMakeTensorUnified(TensorShape & shape) {
	if (shape.count == 0) {
		std::cout << " Shape has invalid count (4th dim) - setting to 1\n";
		shape.count = 1;
	}

	float* m;
	if(cudaMallocManaged(&m, tensorSize(shape) * sizeof(float)) != cudaSuccess) {
		std::cout << "Unable to cudaMallocManaged tensor.\n" ;
		return NULL;
	}

	#ifdef PARALLEL_INIT
	tensorInit(m, shape);
	#else
	uint64_t offset;
	srand(time(NULL));
	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<float> distribution(0.0, 1.0);

	//	Implement NCHW layout
	for (uint32_t count = 0; count < shape.count; ++ count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
					offset = count * shape.channels * shape.width * shape.height + chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					m[offset] = distribution(generator);
				}
			}
		}
	}
	#endif

	return m;
}


float* cudaMakeVector(uint64_t size) {
	float *host = NULL, *device;

	if(makeVector(&host, size) != 0) {
		std::cout << "Unable to make vector.\n" ;
		return NULL;
	}
	if(cudaMalloc(&device, size * sizeof(float)) != cudaSuccess) {
		std::cout << "Unable to cudaMalloc vector.\n" ;
		return NULL;
	}
	if(cudaMemcpy(device, host, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Failed to cudaMemcpy vector.\n");
		return NULL;
	}
	cudaDeviceSynchronize();

	free(host);
	return device;
}


//*****************************************************************************
// CONVOLUTION
//*****************************************************************************
#define CONV_TILE_SIZE 1936 // Size of the shared tile in Bytes (11 x 11 x 4 floats or 22 x 22 x 1 floats)
#define CONV_FILTER_DIM_MAX 22 // Maximum dimension of the convolution filter (per channel)


void runGpuConv (TensorObject* out, TensorObject* inObj, TensorShape fShape, ConvLayerArgs convArgs) {
	TensorShape iShape = inObj->shape;
	TensorShape oShape;

	oShape.count = iShape.count;
	oShape.channels = fShape.count;
	oShape.height = ((iShape.height - fShape.height + convArgs.padH * 2) / convArgs.strideH) + 1;
	oShape.width = ((iShape.width - fShape.width + convArgs.padW * 2) / convArgs.strideW) + 1;
	out->shape = oShape;
	out->tensor = evaluateGpuConv(inObj, fShape, oShape, convArgs);
}


float* evaluateGpuConv (TensorObject* inObj, TensorShape fShape, 
	TensorShape oShape, ConvLayerArgs args) {

	TensorShape iShape = inObj->shape;
	float *in, *out, *filter, *bias;

	// init input tensor
	in = inObj->tensor;
	filter = cudaMakeTensorUnified(fShape);
	bias = cudaMakeVector(oShape.channels);

	if(in == NULL || filter == NULL || bias == NULL) {
		return NULL;
	}

	// Init the ouput tensor
	if(cudaMallocManaged(&out, tensorSize(oShape) * sizeof(float)) != cudaSuccess) {
		std::cout << "Unable to cudaMalloc output tensor.\n" ;
		return NULL;
	}

	// Run Kernel - 1 stream per batch, 1 thread per output element
	assert(fShape.width <= CONV_FILTER_DIM_MAX && fShape.height <= CONV_FILTER_DIM_MAX);
	uint32_t blockWidth = 16;
	uint32_t blockHeight = 16;
	uint32_t numPartials = ceil((float) fShape.channels / (CONV_TILE_SIZE / (sizeof(float) * fShape.width * fShape.height))); 
	dim3 bDim(blockWidth, blockHeight, 1);
	dim3 gDim(ceil((float) oShape.width / blockWidth), ceil((float) oShape.height / blockHeight), oShape.channels * numPartials);
	cudaStream_t streams[oShape.count];

	auto start = std::chrono::high_resolution_clock::now();
	for(uint64_t count = 0; count < oShape.count; count++) {
		float* input = &in[count * iShape.channels * iShape.width * iShape.height];
		float* output = &out[count * oShape.channels * oShape.width * oShape.height];

		cudaStreamCreate(&streams[count]);
		convLayer_gpu<<<gDim, bDim, 0, streams[count]>>>(input, iShape, filter, fShape, bias, output, oShape, args, 1);
	}
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaFree(in);
	cudaFree(filter);
	cudaFree(bias);
	return out;
}


__global__
void convLayer_gpu(float * input, TensorShape iShape,
	float * filter, TensorShape fShape, 
	float * bias, float * output, TensorShape oShape, 
	ConvLayerArgs args, uint32_t batchSize) {

	// Thread Variables
	uint32_t tx = threadIdx.x + blockDim.x * blockIdx.x; // Thread x-index in grid
	uint32_t ty = threadIdx.y + blockDim.y * blockIdx.y; // Thread y-index in grid
	uint32_t tz = threadIdx.z + blockDim.z * blockIdx.z; // Thread z-index in grid
	uint32_t tidxBlk = threadIdx.x + threadIdx.y * blockDim.x; // 1D index of the thread in the block (row-major)
	
	// Block variables
	uint32_t blockDepth = CONV_TILE_SIZE / (sizeof(float) * fShape.width * fShape.height); // Number of filter channels per block for each output element
	uint32_t numPartials = ceilf((float) fShape.channels / blockDepth); // Number of partial blocks that the filter channels were divided into
	uint32_t filterOffset = (tz % numPartials) * blockDepth; // Channel offset for a particular filter block
	uint32_t outChannel = tz / numPartials; // Actual output channel for a filter block
	
	// Tile variables
	uint32_t tileDepth = fShape.channels - filterOffset < blockDepth ? fShape.channels - filterOffset : blockDepth;
	__shared__ float filterTile[CONV_TILE_SIZE];

	// Load the filter weights for this output channel
	uint32_t blkSize = blockDim.x * blockDim.y * blockDim.z;
	uint32_t loadNum = ceilf((float) fShape.height * fShape.width * tileDepth / blkSize);

	for(uint32_t element = tidxBlk; element < tidxBlk + blkSize * loadNum; element += blkSize) {
		if(element < fShape.width * fShape.height * tileDepth) {
			uint32_t m = outChannel * fShape.channels * fShape.width * fShape.height;
			uint32_t k = filterOffset * fShape.width * fShape.height;
			uint32_t filterIdx = m + k + element;

			if(filterIdx < fShape.count * fShape.channels * fShape.width * fShape.height) {
				filterTile[element] = filter[filterIdx];
			}
		}
	}
	__syncthreads();

	if(tx < oShape.width && ty < oShape.height && outChannel < oShape.channels) {
		float outSum = filterOffset == 0 ? bias[outChannel] : 0;

		// Sum part of the output per block
		for(uint32_t k = 0; k < tileDepth; k++) {
			for(uint32_t i = 0; i < fShape.height; i++) {
				for(uint32_t j = 0; j < fShape.width; j++) {
					uint32_t inputChannel = iShape.width * iShape.height * (k + filterOffset);
					int inputY = ty * args.strideH - args.padH + i;
					int inputX = tx * args.strideW - args.padW + j;

					if(inputY >= 0 && inputY < (int) iShape.height && inputX >= 0 && inputX < (int) iShape.width) {
						outSum += input[inputChannel + (inputY * iShape.width) + inputX] * filterTile[k * fShape.width * fShape.height + i * fShape.width + j];
					}
				}
			}
		}

		// Store output
		atomicAdd(&output[outChannel * oShape.height * oShape.width + ty * oShape.width + tx], outSum);
	}
}


//*****************************************************************************
// GEMM
//*****************************************************************************
#define GEMM_TILE_DIM 1024 // Size of the tiles in elements (32 x 32 x 1) was 19 x 19 x 1


void runGpuGemm (TensorObject* out, TensorObject* in, TensorShape weights) {
	uint32_t tileDim = sqrt(GEMM_TILE_DIM);
	TensorShape aShape = in->shape;
	TensorShape bShape = weights;
	TensorShape oShape;
	GemmLayerArgs args = {tileDim, tileDim, (aShape.width + tileDim - 1) / tileDim};

	// Output Initialization
	if (aShape.width != bShape.height || aShape.channels != bShape.channels || aShape.count != bShape.count) {
		std::cout << "Dimensions dont match : " << aShape << " x " << bShape << " \n";
		return;
	}

	oShape.height = aShape.height;
	oShape.width = bShape.width;
	oShape.channels = aShape.channels;
	oShape.count = aShape.count;
	out->shape = oShape;
	out->tensor = evaluateGpuGemm(in, bShape, oShape, args);
}


float* evaluateGpuGemm (TensorObject* aObj, TensorShape bShape, 
		TensorShape oShape, GemmLayerArgs args) {
		
	TensorShape aShape = aObj->shape;
	float *a, *b, *out;

	// Init inputs
	a = aObj->tensor;
	b = cudaMakeTensor(bShape);

	// Allocate output
	if(cudaMallocManaged(&out, tensorSize(oShape) * sizeof(float)) != cudaSuccess) {
		std::cout << "Unable to cudaMalloc output tensor.\n" ;
		return NULL;
	}

	// Run Kernel - 1 thread per output element, supports multiple channels and batches
	dim3 bDim(args.tileW, args.tileH, 1); // Note if tileH/tileW is small effiency will be poor
	dim3 gDim(ceil((float) oShape.width / args.tileW), ceil((float) oShape.height / args.tileH), oShape.channels * args.subTileCount);
	cudaStream_t streams[oShape.count];
	uint32_t tileSize = (bDim.x + 1) * bDim.y * sizeof(float) * 2;

	for(uint64_t count = 0; count < oShape.count; count++) {
		float* aIn = &a[count * aShape.channels * aShape.width * aShape.height];
		float* bIn = &b[count * bShape.channels * bShape.width * bShape.height];
		float* output = &out[count * oShape.channels * oShape.width * oShape.height];

		cudaStreamCreate(&streams[count]);
		gemmLayer_gpu<<<gDim, bDim, tileSize, streams[count]>>>(aIn, aShape, bIn, bShape, output, oShape, args, 1);
	}
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaFree(b);
	return out;
}


__global__
void gemmLayer_gpu (float * a, TensorShape aShape, float * b, TensorShape bShape,
		float * o, TensorShape oShape,
		GemmLayerArgs args, uint32_t batchSize) {

	// Thread Variables
	uint32_t tx = threadIdx.x + blockDim.x * blockIdx.x; // Thread x-index in grid
	uint32_t ty = threadIdx.y + blockDim.y * blockIdx.y; // Thread y-index in grid
	uint32_t tz = threadIdx.z + blockDim.z * blockIdx.z; // Thread z-index in grid
	uint32_t tidxBlk = threadIdx.x + threadIdx.y * blockDim.x; // 1D index of the thread in the block (row-major)

	// Shared Memory Variables
	extern __shared__ float gemmTiles[]; // Shared memory block
	float* aTile = gemmTiles;
	float* bTile = &gemmTiles[blockDim.y * blockDim.x];

	// Tile/Matrix Variables
	uint32_t aSize = aShape.channels * aShape.width * aShape.height;
	uint32_t bSize = bShape.channels * bShape.width * bShape.height;
	uint32_t outChannel = tz / args.subTileCount; // Actual output channel
	uint32_t subTile = tz % args.subTileCount; // Subtile for a given output tile

	// Load local tiles
	uint32_t tileX = threadIdx.x + subTile * blockDim.x;
	uint32_t tileY = threadIdx.y + subTile * blockDim.y;
	uint32_t aIdx = outChannel * aShape.width * aShape.height + (ty * aShape.width) + tileX;
	uint32_t bIdx = outChannel * bShape.width * bShape.height + (tileY * bShape.width) + tx;

	aTile[tidxBlk] = tileX < aShape.width && aIdx < aSize ? a[aIdx] : 0;
	bTile[tidxBlk] = tileY < bShape.height && bIdx < bSize ? b[bIdx] : 0;
	__syncthreads();

	// Bounds check
	if(tx < oShape.width && ty < oShape.height) {	

		// Calculate the partial output
		float partial = 0;
		for(uint32_t element = 0; element < blockDim.y; element++) {
			partial += aTile[threadIdx.y * blockDim.x + element] * bTile[element * blockDim.x+ threadIdx.x];
		}

		atomicAdd(&o[outChannel * oShape.width * oShape.height + ty * oShape.width + tx], partial);
	}
}