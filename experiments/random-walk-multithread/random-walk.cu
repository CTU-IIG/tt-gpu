
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define DEVICE_NUMBER (0)

typedef struct {
    int nofThreads;
    int nofBlocks;
    int32_t nof_repetitions;
    int data_size;
    int buffer_length;
    unsigned int *targetMeasOH;
    unsigned int hostMeasOH;
    int *hostBuffer;
    int *targetBuffer;
    uint64_t *target_realSum;
    uint64_t host_realSum;
    unsigned int *target_times;
    unsigned int *host_times;
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
static int createShuffledArray(int * buffer, int nofElem){

    // Seed random
    srand(time(0));

    // Link sequentially
    for(int i = 0; i< nofElem-1; i++){
        buffer[i] = i+1;
    }
    buffer[nofElem-1] = 0;

    // Shuffle array	
    for (int i = 0; i<nofElem;i++){
        int rndi, tmp1, tmp2, tmp3; 
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
    uint64_t sum = 0;
    start = clock();
    for (int j = 0; j < params.buffer_length; j++){
        sum +=j;
    }
    stop = clock();
    *params.targetMeasOH = ((unsigned int)(stop-start))/params.buffer_length;
    *params.target_realSum = sum;
}

/*
static __global__ void randomWalkSameElement(param_t params) {
    int current;
    unsigned int time_start, time_end, time_acc;
    uint64_t sum;

    int tindex = blockDim.x*blockIdx.x*params.nof_repetitions + params.nof_repetitions *threadIdx.x;
    int curr_start = blockIdx.x%params.buffer_length;

    // Warm up data cache    
    for(int i = 0; i < params.buffer_length; i++){
        sum += params.targetBuffer[i%params.buffer_length];
    }

    // Run experiment multiple times. First iteration (-1) is to warm up icache
    for (int i = -1; i < params.nof_repetitions; i++){

        sum = 0;
        time_acc = 0;
        current = curr_start;

        __syncthreads();
        time_start = clock();
        for(int j = 0; j < params.buffer_length; j++){
            current = params.targetBuffer[current];
            sum += current;
        }
        time_end = clock();
        time_acc += (time_end - time_start);
        __syncthreads();

        *params.target_realSum = sum;

        // Do not write time for warm up iteration       
        if (i>=0){
            // Write element access time with measurement overhead
            params.target_times[tindex+i] = time_acc/params.buffer_length-(*params.targetMeasOH);
        }
    }
}
*/

static __global__ void randomWalkDiffElement(param_t params) {
    int current;
    unsigned int time_start, time_end;
    unsigned int time_acc, oh;
    uint64_t sum;
    oh = *(params.targetMeasOH);

    int tindex = blockDim.x*blockIdx.x*params.nof_repetitions + params.nof_repetitions *threadIdx.x;
    int curr_start = (blockDim.x*blockIdx.x + threadIdx.x)%params.buffer_length;

    // Warm up data cache    
    for(int i = 0; i < params.buffer_length; i++){
        sum += params.targetBuffer[i%params.buffer_length];
    }
    *params.target_realSum = sum;

    // Run experiment multiple times. First iteration (-1) is to warm up icache
    for (int i = -2; i < params.nof_repetitions; i++){

        sum = 0;
        time_acc = 0;
        current = curr_start;

        __syncthreads();

        time_start = clock();
        for(int j = 0; j < params.buffer_length; j++){
            current = params.targetBuffer[current];
            sum += current;
        }
        time_end = clock();
        __syncthreads();

        time_acc = (unsigned int)(time_end - time_start);
        *params.target_realSum = sum;

        // Do not write time for warm up iteration       
        if (i>=0){
            // Write element access time with measurement overhead
            params.target_times[tindex+i] = time_acc/(params.buffer_length)-oh;
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
    if (createShuffledArray(params->hostBuffer, params->buffer_length) != 0) return EXIT_FAILURE;

    //allocate device random buffer
    if (CheckCUDAError(cudaMalloc(&params->targetBuffer, \
                    params->buffer_length*sizeof(int)))) return -1;    
    if (CheckCUDAError(cudaMemcpy(params->targetBuffer, \
                    params->hostBuffer, \
                    params->buffer_length*sizeof(int), \
                    cudaMemcpyHostToDevice))) return -1;

    //allocate device times
    int size_time = params->nof_repetitions \
                    * params->nofThreads \
                    * params->nofBlocks \
                    * sizeof(unsigned int);

    if (CheckCUDAError(cudaMalloc(&params->target_times, \
                    size_time))) return -1;

    //allocate host times
    params->host_times = NULL;
    params->host_times = (unsigned int *) malloc(size_time);
    if (!params->host_times) {
        perror("Failed allocating host_times buffer: ");
        return  -1;
    }
    memset(params->host_times,1, size_time);

    // Allocate device accumulator
    if (CheckCUDAError(cudaMalloc(&params->target_realSum, \
                    sizeof(uint64_t)))) return -1;

    // Allocate device measOH
    if (CheckCUDAError(cudaMalloc(&params->targetMeasOH, \
                    sizeof(unsigned int)))) return -1;

    return 0;
}

static int runTest(param_t *params){
    // Get measurement overhead
    getMeasurementOverhead<<<1,1>>>(*params);
    // Launch kernel
    randomWalkDiffElement<<<params->nofBlocks,params->nofThreads>>>(*params);
    //randomWalkSameElement<<<params->nofBlocks,params->nofThreads>>>(*params);

    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Copyback times
    int size_time = params->nof_repetitions \
                    * params->nofThreads \
                    * params->nofBlocks \
                    * sizeof(unsigned int);

    if (CheckCUDAError(cudaMemcpy(params->host_times, \
                    params->target_times, \
                    size_time, \
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
    if (fprintf(params->fd,"\"nof_repetitions\": \"%d\",\n", params->nof_repetitions)  < 0 ) return -1;
    if (fprintf(params->fd,"\"data_size\": \"%d\",\n", params->data_size)  < 0 ) return -1;
    if (fprintf(params->fd,"\"buffer_length\": \"%d\",\n", params->buffer_length)  < 0 ) return -1;
    if (fprintf(params->fd,"\"real_sum\": \"%llu\",\n", (unsigned long long)params->host_realSum)  < 0 ) return -1;
    if (fprintf(params->fd,"\"exp_sum\": \"%llu\",\n", ((unsigned long long)(params->buffer_length-1)*(unsigned long long)params->buffer_length)/2)  < 0 ) return -1;
    if (fprintf(params->fd,"\"measOH\": \"%u\",\n", params->hostMeasOH)  < 0 ) return -1;

    // Write times
    int size_time = params->nof_repetitions \
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
    cudaFree(params->targetBuffer);
    cudaFree(params->target_times);

    // Free host buffers
    free(params->hostBuffer);
    free(params->host_times);
    return 0;
}

static void PrintUsage(const char *name) {
    printf("Usage: %s <#threads> <#blocks> <# of intervals> <size in KB>"
            "<output JSON file name>\n", name);
}

int main(int argc, char **argv) {

    if (argc != 6) {
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
    params.nofThreads = nof_threads;
    params.nofBlocks = nof_blocks;

    int nof_repetitions = atoi(argv[3]);
    if (nof_repetitions <= 0) {
        printf("More than 0 repetitions need to be used. Got %s repetitions\n", argv[3]);
        return EXIT_FAILURE;
    }

    int data_size = atoi(argv[4]);
    if (data_size <= 0) {
        printf("The buffer must be 1 or more KB. Got %s KB\n", argv[4]);
        return EXIT_FAILURE;
    }


    params.nof_repetitions = nof_repetitions;
    params.data_size = data_size*1024;
    params.buffer_length = data_size*1024/sizeof(int);

    params.fd = NULL;
    params.fd = fopen(argv[5],"w");
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
