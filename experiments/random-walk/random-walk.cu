#include <errno.h>
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define DEVICE_NUMBER (0)
//#define USE_SHARED
#define SHARED_SIZE (8192) // 32kbytes with int
#define SPIN_DURATION (500000000)

typedef struct {
	uint32_t nofThreads;
	uint32_t nofBlocks;
	int32_t nof_repetitions;
	size_t data_size;
	size_t buffer_length;
	uint32_t *targetMeasOH;
	uint32_t hostMeasOH;
	uint32_t *hostBuffer;
	uint32_t *targetBuffer;
	uint64_t *target_realSum;
	uint64_t host_realSum;
	clock_t *target_times;
	clock_t *host_times;
	FILE *fd;
} param_t;

// Prints a message and returns zero if the given value is not cudaSuccess
#define CheckCUDAError(val) (InternalCheckCUDAError((val), #val, __FILE__, __LINE__))

// Called internally by CheckCUDAError
static int InternalCheckCUDAError(cudaError_t result, const char *fn,
		const char *file, int line) {
	if (result == cudaSuccess) return 0;
	printf("CUDA error %d in %s, line %d (%s): %s\n", (int) result, file, line,
			fn, cudaGetErrorString(result));
	return -1;
}
#ifdef USE_SHARED
// Returns the value of CUDA's global nanosecond timer.
static __device__ inline uint64_t GlobalTimer64(void) {
	// Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
	// around after the upper bits have already been read. Work around this by
	// reading the high bits a second time. Use the second value to detect a
	// rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
	// would be valid, it's passed over during the duration of the reading. If no
	// rollover occurred, just return the initial reading.
	volatile uint64_t first_reading;
	volatile uint32_t second_reading;
	uint32_t high_bits_first;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
	high_bits_first = first_reading >> 32;
	asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
	if (high_bits_first == second_reading) {
		return first_reading;
	}
	// Return the value with the updated high bits, but the low bits set to 0.
	return ((uint64_t) second_reading) << 32;
}
#endif

/*! \brief  Create random 32bit number
 *  \return returns random 32bit number
 *  Uses rand() function to create a random 32 bit number using two calls
 */
static uint32_t random32(void){
	return (rand() ^ (rand() << 15));
}

/*! \brief  Create a randomized array for random walks
 *  \param buffer Pointer to allocated memory segment
 *  \param nofElem Number of elements in array
 *  \return returns error
 */
static int createShuffledArray(uint32_t * buffer, size_t nofElem){

	// Seed random
	srand(time(0));

	// Link sequentially
	for(uint32_t i = 0; i< nofElem-1; i++){
		buffer[i] = i+1;
	}
	buffer[nofElem-1] = 0;

	// Shuffle array	
	for (size_t i = 0; i<nofElem;i++){
		size_t rndi, tmp1, tmp2, tmp3; 
		rndi = random32()%nofElem;
		if (rndi == i) continue;	

		tmp1 = buffer[i];
		tmp2 = buffer[rndi];
		tmp3 = buffer[tmp2];
		if (i== tmp2) continue;

		// Reassign links
		buffer[i] = tmp2;
		buffer[rndi] = tmp3;
		buffer[tmp2] = tmp1;
	}
	return 0;
}

static __global__ void getMeasurementOverhead(param_t params) {
	unsigned int start, stop;
	uint64_t sum;

	start = clock();
	for(int j = 0; j < params.buffer_length; j++){
		sum += j;
	}
	stop = clock();

	*params.targetMeasOH = (stop-start)/params.buffer_length;
	*params.target_realSum = sum;
}

#ifdef USE_SHARED
// Uses 8192 bytes of statically-defined shared memory.
static __device__ uint32_t UseSharedMemory(void) {
	__shared__ uint32_t shared_mem_arr[SHARED_SIZE];
	uint32_t num_threads, elts_per_thread, i;
	num_threads = blockDim.x;
	elts_per_thread = SHARED_SIZE / num_threads;
	for (i = 0; i < elts_per_thread; i++) {
		shared_mem_arr[threadIdx.x * elts_per_thread + i] = threadIdx.x;
	}
	return shared_mem_arr[threadIdx.x * elts_per_thread];
}

// Accesses shared memory and spins in a loop until at least
// spin_duration nanoseconds have elapsed.
static __global__ void spinSHM(uint64_t spin_duration) {
	uint32_t shared_mem_res;
	uint64_t start_time = GlobalTimer64();

	// In the shared memory loop, set the value for a window of elements.
	shared_mem_res = UseSharedMemory();
	// The actual spin loop--most of this kernel code is for recording block and
	// kernel times.
	while ((GlobalTimer64() - start_time) < spin_duration) {
		continue;
	}
}

#endif

static __global__ void randomWalk(param_t params) {
	uint32_t current;
	clock_t time_start;
	clock_t time_end;
	clock_t time_acc;
	uint64_t sum;
	clock_t oh = *params.targetMeasOH;

	if (blockIdx.x != 0) return;
	if (threadIdx.x != 0) return;

	// Warm up data cache    
	for(size_t i = 0; i < params.buffer_length; i++){
		sum += params.targetBuffer[i%params.buffer_length];
	}

	// Run experiment multiple times. First iteration (-1) is to warm up icache
	current = 0;
	for (int i = -2; i < params.nof_repetitions; i++){
		sum = 0;
		time_acc = 0;

		time_start = clock();
		for(int j = 0; j < params.buffer_length; j++){
			current = params.targetBuffer[current];
			sum += current;
		}
		time_end = clock();

		time_acc = time_end - time_start;
		*params.target_realSum = sum;

		// Do not write time for warm up iteration       
		if (i>=0){
			// Write element access time with measurement overhead
			params.target_times[i] = (time_acc/params.buffer_length)-oh;
		}
	}
}

static int initializeTest(param_t *params){
	//allocate buffer
	params->hostBuffer = NULL;
	params->hostBuffer = (uint32_t *) malloc(params->buffer_length*sizeof(uint32_t));
	if (!params->hostBuffer) {
		perror("Failed allocating host buffer: ");
		return  -1;
	}
	if (createShuffledArray(params->hostBuffer, params->buffer_length) != 0) return EXIT_FAILURE;

	//allocate device random buffer
	if (CheckCUDAError(cudaMalloc(&params->targetBuffer, \
					params->buffer_length*sizeof(uint32_t)))) return -1;    
	if (CheckCUDAError(cudaMemcpy(params->targetBuffer, \
					params->hostBuffer, \
					params->buffer_length*sizeof(uint32_t), \
					cudaMemcpyHostToDevice))) return -1;

	//allocate device times
	if (CheckCUDAError(cudaMalloc(&params->target_times, \
					params->nof_repetitions*sizeof(clock_t)))) return -1;

	// Allocate device accumulator
	if (CheckCUDAError(cudaMalloc(&params->target_realSum, \
					sizeof(uint64_t)))) return -1;

	// Allocate device measOH
	if (CheckCUDAError(cudaMalloc(&params->targetMeasOH, \
					sizeof(uint32_t)))) return -1;

	//allocate host times
	params->host_times = NULL;
	params->host_times = (clock_t *) malloc(params->nof_repetitions*sizeof(clock_t));
	if (!params->host_times) {
		perror("Failed allocating host_times buffer: ");
		return  -1;
	}
	memset(params->host_times,0, params->nof_repetitions*sizeof(clock_t));

	return 0;
}

static int runTest(param_t *params){
	cudaProfilerStart();
	getMeasurementOverhead<<<1,1>>>(*params);
#ifdef USE_SHARED
	cudaStream_t stream[5];
	for (int i = 0; i < 5; ++i)
		cudaStreamCreate(&stream[i]);
	// Get measurement overhead
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

	// Launch kernel
	spinSHM<<<1,1,0,stream[1]>>>(SPIN_DURATION);
	spinSHM<<<1,1,0,stream[2]>>>(SPIN_DURATION);
	spinSHM<<<1,1,0,stream[3]>>>(SPIN_DURATION);
	spinSHM<<<1,1,0,stream[4]>>>(SPIN_DURATION);
	randomWalk<<<1,1,0,stream[0]>>>(*params);
#else
	randomWalk<<<1,1>>>(*params);
#endif
	// Synchronize with device
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;
#ifdef USE_SHARED
	for (int i = 0; i < 5; ++i) 
		cudaStreamDestroy(stream[i]);
#endif

	// Copyback times
	if (CheckCUDAError(cudaMemcpy(params->host_times, \
					params->target_times, \
					params->nof_repetitions*sizeof(clock_t), \
					cudaMemcpyDeviceToHost))) return -1;

	// Copyback sum
	params->host_realSum=0; 
	if (CheckCUDAError(cudaMemcpy(&params->host_realSum, \
					params->target_realSum, \
					sizeof(uint64_t), \
					cudaMemcpyDeviceToHost))) return -1;

	// Copyback target meas oh
	params->hostMeasOH=0; 
	if (CheckCUDAError(cudaMemcpy(&params->hostMeasOH, \
					params->targetMeasOH, \
					sizeof(uint32_t), \
					cudaMemcpyDeviceToHost))) return -1;
	cudaProfilerStop();
	return 0;
}

static int writeResults(param_t *params){

	if (fprintf(params->fd,"{\n") < 0 ) return -1;
	// Write device info
	cudaDeviceProp deviceProp;
	if (CheckCUDAError(cudaGetDeviceProperties(&deviceProp, DEVICE_NUMBER))) return -1;
	int driverVersion = 0;
	if (CheckCUDAError(cudaDriverGetVersion(&driverVersion))) return -1;
	int runtimeVersion = 0;
	if (CheckCUDAError(cudaRuntimeGetVersion(&runtimeVersion))) return -1;
	if (fprintf(params->fd,"\"driverVer\": \"%d\",\n", driverVersion)  < 0 ) return -1;
	if (fprintf(params->fd,"\"runTimeVer\": \"%d\",\n", runtimeVersion)  < 0 ) return -1;
	if (fprintf(params->fd,"\"clockRate\": \"%d\",\n", deviceProp.clockRate)  < 0 ) return -1;
	if (fprintf(params->fd,"\"globalL1CacheSupported\": \"%d\",\n", deviceProp.globalL1CacheSupported)  < 0 ) return -1;
	if (fprintf(params->fd,"\"localL1CacheSupported\": \"%d\",\n", deviceProp.localL1CacheSupported)  < 0 ) return -1;
	if (fprintf(params->fd,"\"l2CacheSize\": \"%d\",\n", deviceProp.l2CacheSize)  < 0 ) return -1;
	if (fprintf(params->fd,"\"memoryBusWidth\": \"%d\",\n", deviceProp.memoryBusWidth)  < 0 ) return -1;
	if (fprintf(params->fd,"\"memoryClockRate\": \"%d\",\n", deviceProp.memoryClockRate)  < 0 ) return -1;
	if (fprintf(params->fd,"\"multiProcessorCount\": \"%d\",\n", deviceProp.multiProcessorCount)  < 0 ) return -1;
	if (fprintf(params->fd,"\"regsPerBlock\": \"%d\",\n", deviceProp.regsPerBlock)  < 0 ) return -1;
	if (fprintf(params->fd,"\"regsPerMultiprocessor\": \"%d\",\n", deviceProp.regsPerMultiprocessor)  < 0 ) return -1;
	if (fprintf(params->fd,"\"sharedMemPerBlock\": \"%zu\",\n", deviceProp.sharedMemPerBlock)  < 0 ) return -1;
	if (fprintf(params->fd,"\"sharedMemPerMultiprocessor\": \"%zu\",\n", deviceProp.sharedMemPerMultiprocessor)  < 0 ) return -1;
	if (fprintf(params->fd,"\"warpSize\": \"%d\",\n", deviceProp.warpSize)  < 0 ) return -1;

	cudaFuncCache config;
	if (CheckCUDAError(cudaDeviceGetCacheConfig ( &config ) )) return -1;
	if (fprintf(params->fd,"\"cacheConfig\": \"%d\",\n", config)  < 0 ) return -1;

	// Write header
	if (fprintf(params->fd,"\"nofThreads\": \"%u\",\n", params->nofThreads)  < 0 ) return -1;
	if (fprintf(params->fd,"\"nofBlocks\": \"%u\",\n", params->nofBlocks)  < 0 ) return -1;
	if (fprintf(params->fd,"\"nof_repetitions\": \"%d\",\n", params->nof_repetitions)  < 0 ) return -1;
	if (fprintf(params->fd,"\"data_size\": \"%zu\",\n", params->data_size)  < 0 ) return -1;
	if (fprintf(params->fd,"\"buffer_length\": \"%zu\",\n", params->buffer_length)  < 0 ) return -1;
	if (fprintf(params->fd,"\"real_sum\": \"%llu\",\n", (unsigned long long)params->host_realSum)  < 0 ) return -1;
	if (fprintf(params->fd,"\"exp_sum\": \"%lu\",\n", ((params->buffer_length-1)*params->buffer_length)/2)  < 0 ) return -1;
	if (fprintf(params->fd,"\"measOH\": \"%u\",\n", params->hostMeasOH)  < 0 ) return -1;

	// Write times
	if (fprintf(params->fd,"\"times\":[\n") < 0 ) return -1;
	for (int32_t i = 0; i < params->nof_repetitions-1; i++){
		if (fprintf(params->fd,"\"%Lf\",\n",(long double)params->host_times[i]) < 0 ) return -1;
	}
	if (fprintf(params->fd,"\"%Lf\"]\n}", (long double)params->host_times[params->nof_repetitions-1]) < 0 ) return -1;

	if (fclose(params->fd) < 0) return -1;
	return 0;
}

static int cleanUp(param_t *params){
	// Free target buffers
	cudaFree(params->targetBuffer);
	cudaFree(params->target_times);

	// Free host buffers
	free(params->hostBuffer);
	free(params->host_times);
	return 0;
}

static void PrintUsage(const char *name) {
	printf("Usage: %s <# of intervals> <size in KB> <cache mode>"
			"<output JSON file name>\n", name);
}

int main(int argc, char **argv) {

	if (argc != 5) {
		PrintUsage(argv[0]);
		return 1;
	}

	param_t params;

	// Parse input parameter
	int nof_repetitions = atoi(argv[1]);
	if (nof_repetitions <= 0) {
		printf("More than 0 repetitions need to be used. Got %s repetitions\n", argv[2]);
		return EXIT_FAILURE;
	}

	int data_size = atoi(argv[2]);
	if (data_size <= 0) {
		printf("The buffer must be 1 or more KB. Got %s KB\n", argv[3]);
		return EXIT_FAILURE;
	}

	/*
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g6c9cc78ca80490386cf593b4baa35a15

cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
	 */
	cudaFuncCache cacheMode = (cudaFuncCache)atoi(argv[3]);
	if (cacheMode < cudaFuncCachePreferNone || cacheMode > cudaFuncCachePreferEqual) {
		printf("cacheMode must be between 0 and 3. Got %s\n", argv[3]);
		return EXIT_FAILURE;
	}

	params.nof_repetitions = nof_repetitions;
	params.data_size = data_size*1024;
	params.buffer_length = data_size*1024/sizeof(uint32_t);
	params.nofBlocks = 1;
	params.nofThreads = 1;
	params.fd = NULL;
	params.fd = fopen(argv[4],"w");
	if (params.fd == NULL) {
		perror("Error opening output file:");
		return EXIT_FAILURE;
	}

	// Set CUDA device
	if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))) {
		return EXIT_FAILURE;
	}

	// Set cache mode
	if (CheckCUDAError(cudaDeviceSetCacheConfig(cacheMode))) {
		return EXIT_FAILURE;
	}

	if (CheckCUDAError(cudaFuncSetCacheConfig(randomWalk, cacheMode))) {
		return EXIT_FAILURE;
	}

	// Initialize parameters
	if (initializeTest(&params) < 0) return EXIT_FAILURE;

	// Run test
	if (runTest(&params) < 0) return EXIT_FAILURE;

	// Write results
	if (writeResults(&params) < 0){
		perror("Error while writing outpufile: ");
		return EXIT_FAILURE;
	}

	// Clean up
	if (cleanUp(&params) < 0) return EXIT_FAILURE;

	printf("Finished testrun\n");
	cudaDeviceReset();
	return 0;
}

