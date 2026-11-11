
#include "cudaLib.cuh"
#define DEBUG_PRINT_DISABLE

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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
