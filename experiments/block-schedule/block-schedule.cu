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

#define NOF_BLOCKS    (66)
#define NOF_THREADS   (32)
#define TIME_STEP_NS  (1000000000) //5us


typedef struct {
    uint64_t *targetTimes;
    uint64_t *hostTimes;
    unsigned int *targetSmid;
    unsigned int hostSmid[NOF_BLOCKS];
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

static __device__ __inline__ unsigned int get_smid(void)
{
    unsigned int ret;
    asm("mov.u32 %0, %smid;":"=r"(ret) );
    return ret;
}

static __device__ __inline__ uint64_t getTime(void){
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

static __global__ void getGlobalTimerSpin(param_t params) {
    uint64_t mystart = getTime();
    while(getTime() < mystart+TIME_STEP_NS);
    __syncthreads();
    if(threadIdx.x == 0){
        params.targetTimes[blockIdx.x] = mystart;
        params.targetSmid[blockIdx.x] = get_smid();
    }
}


static int initializeTest(param_t *params){
    //allocate buffer
    params->hostTimes = NULL;
    params->hostTimes = (uint64_t *) malloc(NOF_BLOCKS*sizeof(uint64_t));
    if (!params->hostTimes) {
        perror("Failed allocating host buffer: ");
        return  -1;
    }
    
    memset(params->hostTimes,0, NOF_BLOCKS*sizeof(uint64_t));

    //allocate device random buffer
    if (CheckCUDAError(cudaMalloc(&params->targetTimes, \
                    NOF_BLOCKS*sizeof(uint64_t)))) return -1;
    
    if (CheckCUDAError(cudaMalloc(&params->targetSmid, \
                    NOF_BLOCKS*sizeof(unsigned int)))) return -1;
    
    return 0;
}

static int runTest(param_t *params){

    getGlobalTimerSpin<<<NOF_BLOCKS,NOF_THREADS>>>(*params);
    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Copyback times
    if (CheckCUDAError(cudaMemcpy(params->hostTimes, \
                    params->targetTimes, \
                    NOF_BLOCKS*sizeof(uint64_t), \
                    cudaMemcpyDeviceToHost))) return -1;
    if (CheckCUDAError(cudaMemcpy(params->hostSmid, \
                    params->targetSmid, \
                    NOF_BLOCKS*sizeof(unsigned int), \
                    cudaMemcpyDeviceToHost))) return -1;

    return 0;
}

static int writeResults(param_t *params){

    if (fprintf(params->fd,"{\n") < 0 ) return -1;


    // Print blocks
    if (fprintf(params->fd,"\"blocks\":[\n") < 0 ) return -1;
    for(int j = 0; j<NOF_BLOCKS-1;j++){
        if (fprintf(params->fd,"\"%d\",\n", j) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%d\"],\n", NOF_BLOCKS-1) < 0 ) return -1;

    // Print SMID
    for(int j = 0; j<NOF_BLOCKS;j++){
        if (fprintf(params->fd,"\"smid_%d\": \"%d\",\n", j,params->hostSmid[j])  < 0 ) return -1;
    }


    // Write times
    if (fprintf(params->fd,"\"times\":[\n") < 0 ) return -1;

    for (int i = 0; i < NOF_BLOCKS-1; i++){
        if (fprintf(params->fd,"\"%lu\",\n", params->hostTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lu\"]\n", params->hostTimes[NOF_BLOCKS-1]) < 0 ) return -1;
    
    if (fprintf(params->fd,"\n}") < 0 ) return -1;

    if (fclose(params->fd) < 0) return -1;
    return 0;
}

static int cleanUp(param_t *params){
    // Free target buffers
    cudaFree(params->targetTimes);

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
