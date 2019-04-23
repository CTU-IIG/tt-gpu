#include <errno.h>
#include <error.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define DEVICE_NUMBER (0)

#define NOF_KERNEL    (2)
#define NOF_PASSES    (10)
#define START_TIME_OFFSET_NS (1000000000) //1s

typedef struct {
    uint64_t *hostTimes;
    FILE *fd;
} param_t;

typedef struct {
    uint64_t *targetStartTime;
    int64_t *token;
    uint64_t *targetTimes;
} kernel_t;

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

static __device__ __inline__ uint64_t getTime(void){
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

static __global__ void kernel1(kernel_t params) {
    uint64_t reg_start = *params.targetStartTime;
    while(getTime() < reg_start);
    __syncthreads();

    for(int i = 0 ; i < NOF_PASSES; i++){ 
        params.targetTimes[i] = getTime();
        *params.token = i*2;
        while(*params.token == i*2){
            //asm volatile("membar.gl;" : : :);
            asm volatile("": : :"memory");
            continue;
        }
    }
}

static __global__ void kernel2(kernel_t params) {
    uint64_t reg_start = *params.targetStartTime;

    while(getTime() < reg_start);
    __syncthreads();

    for(int i = 0 ; i < NOF_PASSES; i++){ 
        while(*params.token != i*2){
            //asm volatile("membar.gl;" : : :);
            asm volatile("": : :"memory");
            continue;
        }
        *params.token= i*2+1;
        params.targetTimes[i] = getTime();
    }
}

static __global__ void getStartTime(uint64_t *startTime) {
    if(threadIdx.x == 0){
        *startTime = getTime() + START_TIME_OFFSET_NS;
    }
    __syncthreads();
}

static int initializeTest(param_t *params){
    //allocate buffer
    params->hostTimes = NULL;
    params->hostTimes = (uint64_t *) malloc(NOF_KERNEL*NOF_PASSES * sizeof(uint64_t));
    if (!params->hostTimes) {
        perror("Failed allocating host buffer: ");
        return  -1;
    }
    
    memset(params->hostTimes,0, NOF_KERNEL * NOF_PASSES * sizeof(uint64_t));

    return 0;
}

static int runTest(param_t *params){
    int64_t *token;
    if (CheckCUDAError(cudaHostAlloc(&token, sizeof(int64_t), cudaHostAllocMapped))) return -1;
    *token = -100;
    uint64_t *startTime;

    if (CheckCUDAError(cudaMalloc(&startTime, \
                    sizeof(uint64_t)))) return -1;

    getStartTime<<<1,1>>>(startTime);
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    kernel_t kernelData[NOF_KERNEL];
    cudaStream_t stream[NOF_KERNEL];

    for( int i = 0; i< NOF_KERNEL;i++){
        cudaStreamCreate(&stream[i]);
        kernelData[i].targetStartTime = startTime;
        kernelData[i].token = token;
        if (CheckCUDAError(cudaMalloc(&kernelData[i].targetTimes, \
                        NOF_PASSES * sizeof(uint64_t)))) return -1;

    }

    kernel1<<<1,1, 0, stream[0]>>>(kernelData[0]);
    kernel2<<<1,1, 0, stream[1]>>>(kernelData[1]);
    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    for( int i = 0; i< NOF_KERNEL;i++){
        // Copyback times
        if (CheckCUDAError(cudaMemcpy(&params->hostTimes[i*NOF_PASSES], \
                        kernelData[i].targetTimes, \
                        NOF_PASSES*sizeof(uint64_t), \
                        cudaMemcpyDeviceToHost))) return -1;
        cudaFree(kernelData[i].targetTimes);
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(token);
    cudaFree(startTime);

    return 0;
}

static int writeResults(param_t *params){

    if (fprintf(params->fd,"{\n") < 0 ) return -1;

	if (fprintf(params->fd,"\"nofpasses\": \"%d\",\n", NOF_PASSES)  < 0 ) return -1;
	if (fprintf(params->fd,"\"nofkernel\": \"%d\",\n", NOF_KERNEL)  < 0 ) return -1;

    // Write times
    if (fprintf(params->fd,"\"times\":[\n") < 0 ) return -1;

    for (int i = 0; i < NOF_KERNEL*NOF_PASSES-1; i++){
        if (fprintf(params->fd,"\"%lu\",\n", params->hostTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lu\"]\n", params->hostTimes[NOF_KERNEL*NOF_PASSES-1]) < 0 ) return -1;
    
    if (fprintf(params->fd,"\n}") < 0 ) return -1;

    if (fclose(params->fd) < 0) return -1;
    return 0;
}

static int cleanUp(param_t *params){
    // Free host buffers
    free(params->hostTimes);
    return 0;
}

static void PrintUsage(const char *name) {
    printf("Usage: %s <output JSON file name>\n", name);
}

int main(int argc, char **argv) {

    if (argc != 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    param_t params;

    params.fd = NULL;
    params.fd = fopen(argv[1],"w");
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
