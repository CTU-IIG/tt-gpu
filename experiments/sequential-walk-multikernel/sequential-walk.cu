
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define DEVICE_NUMBER (0)
// membar implementation simmilar to https://bigfoot.cs.unc.edu:3000/otternes/cuda_scheduling_examiner/src/master/src/barrier_wait.c
typedef struct {
    int *threadsRemaining;
    int *sense;
    int thread_count;
} barrier_t;

typedef struct {
    barrier_t barrier;
    int *targetBuffer;
    uint64_t *target_realSum;
    unsigned int *targetMeasOH;
    unsigned int *target_times;
    int32_t nof_repetitions;
    int buffer_length;
    cudaStream_t stream;
} kernel_param_t;


typedef struct {
    int nofThreads;
    int nofBlocks;
    int nofKernel;
    int32_t nof_repetitions;
    int data_size;
    int buffer_length;
    unsigned int hostMeasOH;
    int *hostBuffer;
    uint64_t host_realSum;
    unsigned int *targetMeasOH;
    unsigned int *host_times;
    uint64_t *target_realSum;
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

static void createSequentialArrayHost(param_t params){
    // Link sequentially
    for(int i = 0; i < params.buffer_length; i++){
        params.hostBuffer[i]=(i+params.nofThreads*params.nofBlocks)%params.buffer_length;
    }
}

static __global__ void getMeasurementOverhead(param_t params) {
    unsigned int start, stop;
    uint64_t sum = 0;
    start = clock();
    for (int j = 0; j < params.buffer_length; j++){
        sum +=j;
    }
    stop = clock();
    *params.targetMeasOH = (stop-start)/params.buffer_length;
    *params.target_realSum = sum;
}


static __device__ inline int barrierWait(barrier_t barrier, int* local_sense) {
    *local_sense = !(*local_sense);
    int value = atomicSub(barrier.threadsRemaining, 1);

    if(value==1) {

    atomicExch(barrier.threadsRemaining, barrier.thread_count);
    *(barrier.sense) = *local_sense;
    return 1;
    }

    while (*(barrier.sense) != *local_sense){
        asm volatile("membar.gl;" : : :);
        continue;
    }
    return 1;
}

static __global__ void sequentialWalk(kernel_param_t params) {
    int current;
    unsigned int time_start, time_end, time_acc, oh;
    uint64_t sum;
    int local_sense = 0;
    oh = *params.targetMeasOH;


    int tindex = blockDim.x*blockIdx.x*params.nof_repetitions + params.nof_repetitions *threadIdx.x;

    // Warm up data cache    
    for(int i = 0; i < params.buffer_length; i++){
        sum += params.targetBuffer[i%params.buffer_length];
    }

    // Run experiment multiple times. First iteration (-1) is to warm up icache
    for (int i = -2; i < params.nof_repetitions; i++){

        sum = 0;
        time_acc = 0;
        current = (blockDim.x*blockIdx.x + threadIdx.x)%params.buffer_length;

        barrierWait(params.barrier, &local_sense);
        __syncthreads();
        time_start = clock();
        for(int j = 0; j < params.buffer_length; j++){
            current = params.targetBuffer[current];
            sum += current;
        }
        time_end = clock();
        time_acc += (time_end - time_start);


        *params.target_realSum = sum;
        __syncthreads();

        // Do not write time for warm up iteration       
        if (i>=0){
            // Write element access time with measurement overhead
            params.target_times[tindex+i] = time_acc/params.buffer_length-oh;
        }
    }
}

static int initializeTest(param_t *params){
    //allocate buffer
    params->hostBuffer = NULL;
    params->hostBuffer = (int *) malloc(params->buffer_length*sizeof(int));
    if (!params->hostBuffer) {
        perror("Failed allocating host buffer: ");
        return  -1;
    }
    createSequentialArrayHost(params->hostBuffer, params->buffer_length);

    //allocate device times
    int size_time = params->nof_repetitions \
                    * params->nofThreads \
                    * params->nofBlocks \
                    * params->nofKernel \
                    * sizeof(unsigned int);


    //allocate host times
    params->host_times = NULL;
    params->host_times = (unsigned int *) malloc(size_time);
    if (!params->host_times) {
        perror("Failed allocating host_times buffer: ");
        return  -1;
    }
    memset(params->host_times,0, size_time);

    // Allocate device measOH
    if (CheckCUDAError(cudaMalloc(&params->targetMeasOH, \
                    sizeof(unsigned int)))) return -1;
    if (CheckCUDAError(cudaMalloc(&params->target_realSum, \
                        sizeof(uint64_t)))) return -1;

    // Get measurement overhead
    getMeasurementOverhead<<<1,1>>>(*params);
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    return 0;
}

static int runTest(param_t *params){
    // Allocate streams for kernels
    int size_time = params->nof_repetitions \
                    * params->nofThreads \
                    * params->nofBlocks \
                    * sizeof(unsigned int);

    int *threadsRemaining; 
    int *sense;
    if (CheckCUDAError(cudaHostAlloc(&threadsRemaining, sizeof(int), cudaHostAllocMapped))) return -1;    
    if (CheckCUDAError(cudaHostAlloc(&sense, sizeof(int), cudaHostAllocMapped))) return -1;    
    *threadsRemaining=params->nofThreads*params->nofBlocks*params->nofKernel;
    *sense = 0;

    kernel_param_t kernelp[params->nofKernel];

    for (int i = 0; i < params->nofKernel; ++i){
        cudaStreamCreate(&kernelp[i].stream);

        if (CheckCUDAError(cudaMalloc(&kernelp[i].targetBuffer, \
                        params->buffer_length*sizeof(int)))) return -1;    
        if (CheckCUDAError(cudaMemcpy(kernelp[i].targetBuffer, \
                        params->hostBuffer, \
                        params->buffer_length*sizeof(int), \
                        cudaMemcpyHostToDevice))) return -1;

        kernelp[i].barrier.threadsRemaining=threadsRemaining;
        kernelp[i].barrier.sense=sense;
        kernelp[i].barrier.thread_count = params->nofThreads*params->nofBlocks*params->nofKernel;

        kernelp[i].buffer_length = params->buffer_length;
        kernelp[i].targetMeasOH = params->targetMeasOH;
        kernelp[i].nof_repetitions = params->nof_repetitions;

        if (CheckCUDAError(cudaMalloc(&kernelp[i].target_times, \
                        size_time))) return -1;
        // Allocate device accumulator
        if (CheckCUDAError(cudaMalloc(&kernelp[i].target_realSum, \
                        sizeof(uint64_t)))) return -1;
    }

    for (int i = 0; i < params->nofKernel; ++i){
        // Launch kernel
        //randomWalkDiffElement<<<params->nofBlocks,params->nofThreads, 0, kernelp[i].stream>>>(kernelp[i]);
        sequentialWalk<<<params->nofBlocks,params->nofThreads, 0, kernelp[i].stream>>>(kernelp[i]);
    }

    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    for (int i = 0; i < params->nofKernel; ++i){
        cudaStreamDestroy(kernelp[i].stream);

        // Copyback times
        if (CheckCUDAError(cudaMemcpy(&params->host_times[i*params->nof_repetitions*params->nofThreads*params->nofBlocks], \
                        kernelp[i].target_times, \
                        size_time, \
                        cudaMemcpyDeviceToHost))) return -1;

        // Copyback sum
        params->host_realSum=0; 
        if (CheckCUDAError(cudaMemcpy(&params->host_realSum, \
                        kernelp[i].target_realSum, \
                        sizeof(uint64_t), \
                        cudaMemcpyDeviceToHost))) return -1;
        cudaFree(kernelp[i].target_realSum);
        cudaFree(kernelp[i].target_times);
    }


    // Copyback target meas oh
    params->hostMeasOH=0; 
    if (CheckCUDAError(cudaMemcpy(&params->hostMeasOH, \
                    params->targetMeasOH, \
                    sizeof(unsigned int), \
                    cudaMemcpyDeviceToHost))) return -1;
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
    if (fprintf(params->fd,"\"nofKernel\": \"%u\",\n", params->nofKernel)  < 0 ) return -1;
    if (fprintf(params->fd,"\"nof_repetitions\": \"%d\",\n", params->nof_repetitions)  < 0 ) return -1;
    if (fprintf(params->fd,"\"data_size\": \"%d\",\n", params->data_size)  < 0 ) return -1;
    if (fprintf(params->fd,"\"buffer_length\": \"%d\",\n", params->buffer_length)  < 0 ) return -1;
    if (fprintf(params->fd,"\"real_sum\": \"%llu\",\n", (unsigned long long)params->host_realSum)  < 0 ) return -1;
    if (fprintf(params->fd,"\"exp_sum\": \"%llu\",\n", ((unsigned long long)(params->buffer_length-1)*(unsigned long long)params->buffer_length)/2)  < 0 ) return -1;
    if (fprintf(params->fd,"\"measOH\": \"%u\",\n", params->hostMeasOH)  < 0 ) return -1;

    // Write times
    int size_time = params->nof_repetitions \
                    * params->nofKernel \
                    * params->nofThreads \
                    * params->nofBlocks;

    if (fprintf(params->fd,"\"times\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%u\",\n",params->host_times[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%u\"]\n}", params->host_times[size_time-1]) < 0 ) return -1;

    if (fclose(params->fd) < 0) return -1;
    return 0;
}

static int cleanUp(param_t *params){
    // Free target buffers
    cudaFree(params->targetMeasOH);

    // Free host buffers
    free(params->hostBuffer);
    free(params->host_times);
    return 0;
}

static void PrintUsage(const char *name) {
    printf("Usage: %s <#threads> <#blocks> <# kernel> <# of intervals> <size in KB>"
            "<output JSON file name>\n", name);
}

int main(int argc, char **argv) {

    if (argc != 7) {
        PrintUsage(argv[0]);
        return 1;
    }

    param_t params;

    // Parse input parameter
    int nof_threads = atoi(argv[1]);
    if (nof_threads <= 0) {
        printf("Min one thread. Got %s threads\n", argv[1]);
        return EXIT_FAILURE;
    }

    int nof_blocks = atoi(argv[2]);
    if (nof_blocks <= 0) {
        printf("Min 1 block. Got %s blocks\n", argv[2]);
        return EXIT_FAILURE;
    }

    int nof_kernel = atoi(argv[3]);
    if (nof_kernel <= 0) {
        printf("Min 1 kernel. Got %s blocks\n", argv[2]);
        return EXIT_FAILURE;
    }

    params.nofThreads = nof_threads;
    params.nofBlocks = nof_blocks;
    params.nofKernel = nof_kernel;

    int nof_repetitions = atoi(argv[4]);
    if (nof_repetitions <= 0) {
        printf("More than 0 repetitions need to be used. Got %s repetitions\n", argv[3]);
        return EXIT_FAILURE;
    }

    int data_size = atoi(argv[5]);
    if (data_size <= 0) {
        printf("The buffer must be 1 or more KB. Got %s KB\n", argv[4]);
        return EXIT_FAILURE;
    }


    params.nof_repetitions = nof_repetitions;
    params.data_size = data_size*1024;
    params.buffer_length = data_size*1024/sizeof(int);

    params.fd = NULL;
    params.fd = fopen(argv[6],"w");
    if (params.fd == NULL) {
        perror("Error opening output file:");
        return EXIT_FAILURE;
    }

    // Set CUDA device
    if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))) {
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
