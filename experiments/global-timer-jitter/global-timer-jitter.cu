#include <errno.h>
#include <error.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define DEVICE_NUMBER (0)

#define NOF_STAMPS    (4096) //32kBytes for uint64_t
#define NOF_BLOCKS    (4)
#define SPIN() // spin(10000)

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

static __device__ inline void spin(unsigned int spin_duration) {
    unsigned int start_time = clock();
    while ((clock() - start_time) < spin_duration) {
        continue;
    }
}


static __global__ void getGlobalTimerJitter(param_t params) {
    __shared__ uint64_t times[NOF_STAMPS];
    uint64_t tmp;
    for(int i = -32; i<NOF_STAMPS; i++){
        SPIN();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tmp));
        if(i>=0){
            times[i] = tmp;
        }
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
    params->hostTimes = (uint64_t *) malloc(NOF_BLOCKS*NOF_STAMPS*sizeof(uint64_t));
    if (!params->hostTimes) {
        perror("Failed allocating host buffer: ");
        return  -1;
    }
    memset(params->hostTimes,0, NOF_BLOCKS*NOF_STAMPS*sizeof(uint64_t));

    //allocate device random buffer
    if (CheckCUDAError(cudaMalloc(&params->targetTimes, \
                    NOF_BLOCKS*NOF_STAMPS*sizeof(uint64_t)))) return -1;
    
    if (CheckCUDAError(cudaMalloc(&params->targetSmid, \
                    NOF_BLOCKS*sizeof(unsigned int)))) return -1;

    return 0;
}

static int runTest(param_t *params){

    getGlobalTimerJitter<<<NOF_BLOCKS,1>>>(*params);
    // Synchronize with device
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Copyback times
    if (CheckCUDAError(cudaMemcpy(params->hostTimes, \
                    params->targetTimes, \
                    NOF_BLOCKS*NOF_STAMPS*sizeof(uint64_t), \
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

    for(int j = 0; j<NOF_BLOCKS-1;j++){
        if (fprintf(params->fd,"\"times_%d\":[\n", j) < 0 ) return -1;
        for (int i = 0; i < NOF_STAMPS-1; i++){
            if (fprintf(params->fd,"\"%" PRIu64 "\",\n", params->hostTimes[j*NOF_STAMPS+i]) < 0 ) return -1;
        }
        if (fprintf(params->fd,"\"%" PRIu64 "\"],\n", params->hostTimes[j*NOF_STAMPS+NOF_STAMPS-1]) < 0 ) return -1;
    }
    
    if (fprintf(params->fd,"\"times_%d\":[\n",NOF_BLOCKS-1) < 0 ) return -1;
    for (int i = 0; i < NOF_STAMPS-1; i++){
        if (fprintf(params->fd,"\"%" PRIu64 "\",\n", params->hostTimes[(NOF_BLOCKS-1)*NOF_STAMPS+i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%" PRIu64 "\"]\n}", params->hostTimes[(NOF_BLOCKS-1)*NOF_STAMPS+NOF_STAMPS-1]) < 0 ) return -1;

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
