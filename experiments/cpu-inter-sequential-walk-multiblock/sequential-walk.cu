#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "sequential-walk.h"

#define DEVICE_NUMBER (0)

typedef struct {
    int *targetBuffer;
    uint64_t *target_realSum;
    unsigned int *targetMeasOH;
    unsigned int *target_times;
    int32_t nof_repetitions;
    int buffer_length;
    cudaStream_t stream;
} kernel_param_t;


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
        params.hostBuffer[i]=i;
    }
}

static __global__ void getMeasurementOverhead(param_t params) {
    unsigned int start, stop;
    uint64_t sum = 0;
    start = clock();
    /*
    for (int j = 0; j < params.buffer_length; j++){
        sum +=j;
    }*/
    __syncthreads();
    stop = clock();
    *params.targetMeasOH = (stop-start);
    *params.target_realSum = sum;
}

static __global__ void sequentialWalk(kernel_param_t params) {
    unsigned int time_start, time_end, time_acc, oh;
    uint64_t sum=0;
    oh = *params.targetMeasOH;

    int tindex = blockIdx.x;    
    time_acc = 0;

    __syncthreads();
    if(threadIdx.x == 0){
        time_start = clock();
    }

    //int itercount = 0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < params.buffer_length; j += blockDim.x * gridDim.x) 
    //for (int j = blockIdx.x + gridDim.x; j < params.buffer_length; j += blockDim.x * gridDim.x) 
    {
        if(blockIdx.x%2){
            sum += params.targetBuffer[j];
        }else{
            params.targetBuffer[j] = j;
        }
        //itercount++;
    }
    __syncthreads();
    if (threadIdx.x == 0){
        time_end = clock();
        time_acc += (time_end - time_start);
        *params.target_realSum = sum;

        // Do not write time for warm up iteration       
        // Write element access time with measurement overhead
        params.target_times[tindex] = time_acc-oh;
    }
}

int initializeTest(param_t *params){

    // Set CUDA device
    if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))) {
        return EXIT_FAILURE;
    }

    //allocate buffer
    params->hostBuffer = NULL;
    params->hostBuffer = (int *) malloc(params->buffer_length*sizeof(int));
    if (!params->hostBuffer) {
        perror("Failed allocating host buffer: ");
        return  -1;
    }
    createSequentialArrayHost(*params);

    //allocate device times
    int size_time =  params->nofBlocks \
                    * params->nof_repetitions \
                    * sizeof(unsigned int);


    //allocate host times
    params->host_times = NULL;
    params->host_times = (unsigned int *) malloc(size_time);
    if (!params->host_times) {
        perror("Failed allocating host_times buffer: ");
        return  -1;
    }
    memset(params->host_times,0, size_time);


    params->kernelTimes = NULL;
    params->kernelTimes = (float *) malloc(params->nof_repetitions*sizeof(float));
    if (!params->kernelTimes) {
        perror("Failed allocating host_times buffer: ");
        return  -1;
    }
    memset(params->kernelTimes,0, params->nof_repetitions*sizeof(float));

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

int runTest(param_t *params){
    // Allocate streams for kernels
    int size_time = params->nofBlocks \
                    * sizeof(unsigned int);


    kernel_param_t kernelp;


    if (CheckCUDAError(cudaMalloc(&kernelp.targetBuffer, \
                    params->buffer_length*sizeof(int)))) return -1;    
    if (CheckCUDAError(cudaMemcpy(kernelp.targetBuffer, \
                    params->hostBuffer, \
                    params->buffer_length*sizeof(int), \
                    cudaMemcpyHostToDevice))) return -1;

    kernelp.buffer_length = params->buffer_length;
    kernelp.targetMeasOH = params->targetMeasOH;
    kernelp.nof_repetitions = params->nof_repetitions;

    if (CheckCUDAError(cudaMalloc(&kernelp.target_times, \
                    size_time))) return -1;
    // Allocate device accumulator
    if (CheckCUDAError(cudaMalloc(&kernelp.target_realSum, \
                    sizeof(uint64_t)))) return -1;

    for(int i = -32; i < params->nof_repetitions; i++){
        // Launch kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        sequentialWalk<<<params->nofBlocks,params->nofThreads>>>(kernelp);
        cudaEventRecord(stop);


        // Synchronize with device
        //if (CheckCUDAError(cudaDeviceSynchronize())) return -1;
        cudaEventSynchronize(stop);
        float milliseconds = 0;

        cudaEventElapsedTime(&milliseconds, start, stop);
        if(i>=0){
            printf("Elapsed time: %fms\n",milliseconds);
            params->kernelTimes[i]=milliseconds;


            // Copyback times
            if (CheckCUDAError(cudaMemcpy(&params->host_times[i*params->nofBlocks], \
                            kernelp.target_times, \
                            size_time, \
                            cudaMemcpyDeviceToHost))) return -1;
        }
    }

    // Copyback sum
    params->host_realSum=0; 
    if (CheckCUDAError(cudaMemcpy(&params->host_realSum, \
                    kernelp.target_realSum, \
                    sizeof(uint64_t), \
                    cudaMemcpyDeviceToHost))) return -1;
    cudaFree(kernelp.target_realSum);
    cudaFree(kernelp.target_times);


    // Copyback target meas oh
    params->hostMeasOH=0; 
    if (CheckCUDAError(cudaMemcpy(&params->hostMeasOH, \
                    params->targetMeasOH, \
                    sizeof(unsigned int), \
                    cudaMemcpyDeviceToHost))) return -1;
    return 0;
}

int writeResults(param_t *params){

    if (fprintf(params->fd,"{\n") < 0 ) return -1;
    // Write device info
    cudaDeviceProp deviceProp;
    if (CheckCUDAError(cudaGetDeviceProperties(&deviceProp, DEVICE_NUMBER))) return -1;
    int driverVersion = 0;
    if (CheckCUDAError(cudaDriverGetVersion(&driverVersion))) return -1;
    int runtimeVersion = 0;
    if (CheckCUDAError(cudaRuntimeGetVersion(&runtimeVersion))) return -1;
    if (fprintf(params->fd,"\"clockRate\": \"%d\",\n", deviceProp.clockRate)  < 0 ) return -1;

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
    int size_time = params->nofBlocks * params->nof_repetitions;

    if (fprintf(params->fd,"\"blocktimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%u\",\n",params->host_times[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%u\"],\n", params->host_times[size_time-1]) < 0 ) return -1;

    size_time = params->nof_repetitions;
    if (fprintf(params->fd,"\"kerneltimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%f\",\n",params->kernelTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%f\"]\n}", params->kernelTimes[size_time-1]) < 0 ) return -1;


    if (fclose(params->fd) < 0) return -1;
    return 0;
}

int cleanUp(param_t *params){
    // Free target buffers
    cudaFree(params->targetMeasOH);

    // Free host buffers
    free(params->hostBuffer);
    free(params->kernelTimes);
    free(params->host_times);

    cudaDeviceReset();
    return 0;
}
