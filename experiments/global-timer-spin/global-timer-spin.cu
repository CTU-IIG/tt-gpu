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

#define NOF_STAMPS    (16) //(8192) //32kb smh with unsinged int
#define NOF_BLOCKS    (4)
#define TIME_STEP_NS  (5000) //5us
#define START_INC_NS  (1000000000) //1s


typedef struct {
    unsigned int *targetTimes;
    unsigned int *hostTimes;
    uint64_t *targetStartTime;
    uint64_t hostStartTime;
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

static __global__ void getStartTime(param_t params) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(*params.targetStartTime));
}

static __global__ void getGlobalTimerSpin(param_t params) {
    __shared__ unsigned int times[NOF_STAMPS];
    uint64_t mystart = *params.targetStartTime;
    
    for(uint64_t i = 0; i<NOF_STAMPS; i++){
        while(getTime() < mystart+( i*TIME_STEP_NS ) );
        times[i] = clock();
    }

    // Store times to global memory
    for(int i = 0; i<NOF_STAMPS; i++){
        params.targetTimes[i+blockIdx.x*NOF_STAMPS] = times[i];
    }

    params.targetSmid[blockIdx.x] = get_smid();
}


static int initializeTest(param_t *params){
    //allocate buffer
    params->hostTimes = NULL;
    params->hostTimes = (unsigned int *) malloc(NOF_BLOCKS*NOF_STAMPS*sizeof(unsigned int));
    if (!params->hostTimes) {
        perror("Failed allocating host buffer: ");
        return  -1;
    }
    
    memset(params->hostTimes,0, NOF_BLOCKS*NOF_STAMPS*sizeof(unsigned int));
    params->hostStartTime = 0;

    //allocate device random buffer
    if (CheckCUDAError(cudaMalloc(&params->targetTimes, \
                    NOF_BLOCKS*NOF_STAMPS*sizeof(unsigned int)))) return -1;
    
    if (CheckCUDAError(cudaMalloc(&params->targetSmid, \
                    NOF_BLOCKS*sizeof(unsigned int)))) return -1;
    
    if (CheckCUDAError(cudaMalloc(&params->targetStartTime, \
                    sizeof(uint64_t)))) return -1;

    return 0;
}

static int runTest(param_t *params){

    getStartTime<<<1,1>>>(*params);
    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Increment start time
    if (CheckCUDAError(cudaMemcpy(&params->hostStartTime, \
                    params->targetStartTime, \
                    sizeof(uint64_t), \
                    cudaMemcpyDeviceToHost))) return -1;
    params->hostStartTime= params->hostStartTime+START_INC_NS;
    if (CheckCUDAError(cudaMemcpy(params->targetStartTime, \
                    &params->hostStartTime, \
                    sizeof(uint64_t), \
                    cudaMemcpyHostToDevice))) return -1;
    
    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    getGlobalTimerSpin<<<NOF_BLOCKS,1>>>(*params);
    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Copyback times
    if (CheckCUDAError(cudaMemcpy(params->hostTimes, \
                    params->targetTimes, \
                    NOF_BLOCKS*NOF_STAMPS*sizeof(unsigned int), \
                    cudaMemcpyDeviceToHost))) return -1;
    if (CheckCUDAError(cudaMemcpy(params->hostSmid, \
                    params->targetSmid, \
                    NOF_BLOCKS*sizeof(unsigned int), \
                    cudaMemcpyDeviceToHost))) return -1;

    return 0;
}

static int writeResults(param_t *params){

    if (fprintf(params->fd,"{\n") < 0 ) return -1;


    cudaDeviceProp deviceProp;
    if (CheckCUDAError(cudaGetDeviceProperties(&deviceProp, DEVICE_NUMBER))) return -1;
    if (fprintf(params->fd,"\"clockRatekHz\": \"%d\",\n", deviceProp.clockRate)  < 0 ) return -1;
    if (fprintf(params->fd,"\"stepns\": \"%d\",\n", TIME_STEP_NS)  < 0 ) return -1;

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

    for(int j = 0; j<NOF_BLOCKS-1;j++){
        if (fprintf(params->fd,"\"times_%d\":[\n", j) < 0 ) return -1;
        for (int i = 0; i < NOF_STAMPS-1; i++){
            if (fprintf(params->fd,"\"%u\",\n", params->hostTimes[j*NOF_STAMPS+i]) < 0 ) return -1;
        }
        if (fprintf(params->fd,"\"%u\"],\n", params->hostTimes[j*NOF_STAMPS+NOF_STAMPS-1]) < 0 ) return -1;
    }
    
    if (fprintf(params->fd,"\"times_%d\":[\n",NOF_BLOCKS-1) < 0 ) return -1;

    for (int i = 0; i < NOF_STAMPS-1; i++){
        if (fprintf(params->fd,"\"%u\",\n", params->hostTimes[(NOF_BLOCKS-1)*NOF_STAMPS+i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%u\"]\n}", params->hostTimes[(NOF_BLOCKS-1)*NOF_STAMPS+NOF_STAMPS-1]) < 0 ) return -1;

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
