#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "2DConvolution.h"

#define DEVICE_NUMBER (0)

typedef struct {
    float *A;
    float *B;
    int ni;
    int nj;
    uint64_t *targetTimes;
    unsigned int *smid;
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
} kernel_data_t;


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

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f


static float absVal(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}

static float percentDiff(double val1, double val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
    		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
}

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
static void compareResults(int ni, int nj, float *B, float *BGPU)
{
	int i, j, fail;
	fail = 0;
	
	// Compare outputs from CPU and GPU
	for (i=1; i < (ni-1); i++) 
	{
		for (j=1; j < (nj-1); j++) 
		{
			if (percentDiff(B[i*nj+j], BGPU[i*nj+j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
				printf("fail i: %d j: %d\n", i, j);
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
	
}

static void conv2DCPU(int ni, int nj, float* A, float *B)
{
	int i, j;
	float c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < ni - 1; ++i) // 0
	{
		for (j = 1; j < nj - 1; ++j) // 1
		{
        B[i * nj + j] =  c11 * A[(i - 1) * nj + (j - 1)] + 
                         c21 * A[(i - 1) * nj + (j + 0)] + 
                         c31 * A[(i - 1) * nj + (j + 1)] + 
                         c12 * A[(i + 0) * nj + (j - 1)] + 
                         c22 * A[(i + 0) * nj + (j + 0)] + 
                         c32 * A[(i + 0) * nj + (j + 1)] + 
                         c13 * A[(i + 1) * nj + (j - 1)] + 
                         c23 * A[(i + 1) * nj + (j + 0)] + 
                         c33 * A[(i + 1) * nj + (j + 1)];
		}
	}
}

static void initA(int ni, int nj, float* A)
{
    int i, j;

    for (i = 0; i < ni; ++i)
    {
        for (j = 0; j < nj; ++j)
        {
            A[nj*i+j] = (float)rand()/RAND_MAX;
        }
    }
}

static __device__ __inline__ uint64_t getTime(void){
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

static __global__ void getMeasurementOverhead(param_t data) {
    uint64_t start, stop;
    if(threadIdx.x == 0){
        start = getTime();
    }
    __syncthreads();
    if(threadIdx.x == 0){
        stop = getTime();
        *data.targetMeasOH = stop-start;
    }
}

static __device__ __inline__ unsigned int get_smid(void)
{
    unsigned int ret;
    asm("mov.u32 %0, %%smid;":"=r"(ret) );
    return ret;
}
//#define PREM_SHM_SIZE (32768/(2*sizeof(float))) // Each kernel has 32kBytes of SHM ni=4 nj=1024 inputdata: 4096x4090 
#define PREM_SHM_SIZE (32768/(4*sizeof(float))) // Each kernel has 32kBytes of SHM ni=4, nj=512 inputdata: 4098 4082, 1026x1022
#define PREM_NJ_TILE_SIZE ()
#define PREM_NI_TILE_SIZE ()

// Premification for datasets of size 512x512, 1024x1024 and 4096x4090
static __global__ void convolution2D_kernelPREM(kernel_data_t data)
{
    __shared__ float A_SHM[PREM_SHM_SIZE];
    __shared__ float B_SHM[PREM_SHM_SIZE];
    int prefetch = 0;
    int writeback = 0;

    uint64_t start_time = getTime();
    if(threadIdx.x == 0){
        data.targetTimes[blockIdx.x*2] = start_time;
        data.smid[blockIdx.x] = get_smid();
        printf("Block %d start\n",blockIdx.x);
    }
    __syncthreads();

    float *A = data.A;
    float *B = data.B;

    int ni = data.ni;
    int nj = data.nj;

    int ni_tile_size = 4;
    int nj_tile_size = PREM_SHM_SIZE/ni_tile_size;

    float c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
    c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
    c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
    
    for(int niTile = blockIdx.y; niTile <= ni-4; niTile += 2){
        for(int njTile = blockIdx.x*(nj_tile_size-2); njTile <= nj-nj_tile_size; njTile += (nj_tile_size-2)*gridDim.x){
            
            if(threadIdx.x == 0){
                prefetch++;
                //printf("Block %d niTile: %d, njTile: %d\n",blockIdx.x, niTile, njTile);
            }
            
            // Prefetch data
            for (int i = threadIdx.y; (i < ni_tile_size) && (i + niTile < ni); i += blockDim.y){
                for (int j = threadIdx.x; (j < nj_tile_size) && (j + njTile < nj); j += blockDim.x) {
                                A_SHM[i * nj_tile_size + j] = A[(i+niTile)*nj + (j+njTile)];
                }
            }

            __syncthreads();
            // Compute on SHM
            for (int i = threadIdx.y+1; i < ni_tile_size-1; i += blockDim.y){
                for (int j = threadIdx.x+1; j < nj_tile_size-1; j += blockDim.x) {
                                B_SHM[i * nj_tile_size + j] = c11 * A_SHM[(i - 1) * nj_tile_size + (j - 1)] + 
                                                              c21 * A_SHM[(i - 1) * nj_tile_size + (j + 0)] + 
                                                              c31 * A_SHM[(i - 1) * nj_tile_size + (j + 1)] + 
                                                              c12 * A_SHM[(i + 0) * nj_tile_size + (j - 1)] + 
                                                              c22 * A_SHM[(i + 0) * nj_tile_size + (j + 0)] + 
                                                              c32 * A_SHM[(i + 0) * nj_tile_size + (j + 1)] + 
                                                              c13 * A_SHM[(i + 1) * nj_tile_size + (j - 1)] + 
                                                              c23 * A_SHM[(i + 1) * nj_tile_size + (j + 0)] + 
                                                              c33 * A_SHM[(i + 1) * nj_tile_size + (j + 1)];

                }
            }

            if(threadIdx.x == 0){
                writeback++;
            }
            __syncthreads();
            // Write back data
            for (int i = threadIdx.y+1; (i < ni_tile_size-1) && (i + niTile < ni-1); i += blockDim.y){
                for (int j = threadIdx.x+1; (j < nj_tile_size-1) && (j + njTile < nj-1); j += blockDim.x) {
                                B[(i+niTile) * nj + (j+njTile)] = B_SHM[i*nj_tile_size + j];

                }
            }

            __syncthreads();

        }
    }

    __syncthreads();
    if(threadIdx.x == 0){
        data.targetTimes[blockIdx.x*2+1] = getTime();
        printf("Block %d end, prefetchcount: %d, writebackcount: %d\n",blockIdx.x, prefetch, writeback);
    }
}


static __global__ void convolution2D_kernelLegacy(kernel_data_t data)
{
    uint64_t start_time = getTime();
    if(threadIdx.x == 0){
        data.targetTimes[blockIdx.x*2] = start_time;
        data.smid[blockIdx.x] = get_smid();
    }
    __syncthreads();

    float *A = data.A;
    float *B = data.B;

    int ni = data.ni;
    int nj = data.nj;

    float c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
    c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
    c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	for (int i = blockIdx.y * blockDim.y + threadIdx.y+1; i < ni-1; i += blockDim.y * gridDim.y){
		for (int j = blockIdx.x * blockDim.x + threadIdx.x+1; j < nj-1; j += blockDim.x * gridDim.x) {
                        B[i * nj + j] =  c11 * A[(i - 1) * nj + (j - 1)] + 
                            c21 * A[(i - 1) * nj + (j + 0)] + 
                            c31 * A[(i - 1) * nj + (j + 1)] + 
                            c12 * A[(i + 0) * nj + (j - 1)] + 
                            c22 * A[(i + 0) * nj + (j + 0)] + 
                            c32 * A[(i + 0) * nj + (j + 1)] + 
                            c13 * A[(i + 1) * nj + (j - 1)] + 
                            c23 * A[(i + 1) * nj + (j + 0)] + 
                            c33 * A[(i + 1) * nj + (j + 1)];

		}
	}
    __syncthreads();
    if(threadIdx.x == 0){
        data.targetTimes[blockIdx.x*2+1] = getTime();
    }
}

int initializeTest(param_t *params){

    // Set CUDA device
    if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))) {
        return EXIT_FAILURE;
    }

    //Fill matrix A on host
    float *A= NULL;
    A = (float *) malloc(params->ni*params->nj*sizeof(float));
    if (!A) {
        perror("Failed allocating A matrix on host: ");
        return  -1;
    }
    float *B= NULL;
    B = (float *) malloc(params->ni*params->nj*sizeof(float));
    if (!B) {
        perror("Failed allocating A matrix on host: ");
        return  -1;
    }
    initA(params->ni, params->nj, A);
	conv2DCPU(params->ni, params->nj, A, B);
    //memcpy(B, A, params->ni*params->nj*sizeof(float));
	params->BCPU = B;

	params->BGPU=NULL;
    params->BGPU = (float *) malloc(params->ni*params->nj*sizeof(float));
    if (!params->BGPU) {
        perror("Failed allocating A matrix on host: ");
        return  -1;
    }

    // Allocate kerneldata
    kernel_data_t *kernelData = NULL;
    kernelData = (kernel_data_t*)malloc(params->nofKernels*sizeof(kernel_data_t));
    if (!kernelData) {
        perror("Failed allocating kerneldata: ");
        return  -1;
    }
    memset(kernelData, 0, params->nofKernels*sizeof(kernel_data_t));

    // Fill kernel_data structures
    for(int i = 0; i< params->nofKernels; i++){
        cudaStreamCreate(&kernelData[i].stream);

        // Fill Matrix A
        if (CheckCUDAError(cudaMalloc(&kernelData[i].A, \
                        params->ni*params->nj*sizeof(float)))) return -1;    
        if (CheckCUDAError(cudaMemcpy(kernelData[i].A, \
                        A, \
                        params->ni*params->nj*sizeof(float), \
                        cudaMemcpyHostToDevice))) return -1;
        // Create Matrix B
        if (CheckCUDAError(cudaMalloc(&kernelData[i].B, \
                        params->ni*params->nj*sizeof(float)))) return -1;

        // Allocate time array
        if (CheckCUDAError(cudaMalloc(&kernelData[i].targetTimes, \
                        params->nofBlocks*2*sizeof(uint64_t)))) return -1;

        // Allocate smid space
        if (CheckCUDAError(cudaMalloc(&kernelData[i].smid, \
                        params->nofBlocks*sizeof(unsigned int)))) return -1;

        kernelData[i].ni = params->ni;
        kernelData[i].nj = params->nj;
        cudaEventCreate(&kernelData[i].start);
        cudaEventCreate(&kernelData[i].stop);

    }


    //allocate host times
    params->blockTimes = NULL;
    params->blockTimes = (uint64_t *) malloc(params->nof_repetitions*2*params->nofBlocks*params->nofKernels*sizeof(uint64_t));
    if (!params->blockTimes) {
        perror("Failed allocating hostTimes buffer: ");
        return  -1;
    }
    memset(params->blockTimes,0, params->nof_repetitions*2*params->nofBlocks*params->nofKernels*sizeof(uint64_t));

    // allocate host kernel times for cuda events
    params->kernelTimes = NULL;
    params->kernelTimes = (float *) malloc(params->nofKernels*params->nof_repetitions*sizeof(float));
    if (!params->kernelTimes) {
        perror("Failed allocating kernelTimes buffer: ");
        return  -1;
    }
    memset(params->kernelTimes,0, params->nofKernels*params->nof_repetitions*sizeof(float));

    // allocate host smid buffer
    params->smid = NULL;
    params->smid = (unsigned int *) malloc(params->nofKernels*params->nofBlocks*params->nof_repetitions*sizeof(unsigned int));
    if (!params->smid) {
        perror("Failed allocating smid buffer: ");
        return  -1;
    }
    memset(params->smid,0 , params->nofKernels*params->nofBlocks*params->nof_repetitions*sizeof(float));

    // Allocate device measOH
    if (CheckCUDAError(cudaMalloc(&params->targetMeasOH, \
                    sizeof(unsigned int)))) return -1;

    // Get measurement overhead
    getMeasurementOverhead<<<1,1>>>(*params);
    if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

    // Copyback target meas oh
    params->hostMeasOH=0; 
    if (CheckCUDAError(cudaMemcpy(&params->hostMeasOH, \
                    params->targetMeasOH, \
                    sizeof(unsigned int), \
                    cudaMemcpyDeviceToHost))) return -1;

    // Assigne kernel data
    params->kernelData = (void*) kernelData;

    if(A != NULL) free(A);

    return 0;
}

int runTest(param_t *params){

    kernel_data_t *kernelData = (kernel_data_t*)params->kernelData;

    for(int rep = 0; rep < params->nof_repetitions; rep++){

        for(int kernel = 0; kernel < params->nofKernels; kernel++){
            cudaEventRecord(kernelData[kernel].start, 
                    kernelData[kernel].stream);

            convolution2D_kernelPREM<<<params->nofBlocks,\
                params->nofThreads,\
                0,\
                kernelData[kernel].stream>>>(kernelData[kernel]);

            cudaEventRecord(kernelData[kernel].stop,
                    kernelData[kernel].stream);

        }


        // Synchronize with device
        if (CheckCUDAError(cudaDeviceSynchronize())) return -1;


        for(int kernel = 0; kernel < params->nofKernels; kernel++){
            float milliseconds = 0;

            cudaEventElapsedTime(&milliseconds,\
                    kernelData[kernel].start,\
                    kernelData[kernel].stop);

		// Copyback B and check

        	if (CheckCUDAError(cudaMemcpy(params->BGPU, \
                                kernelData[kernel].B, \
                                params->ni*params->nj*sizeof(float), \
                                cudaMemcpyDeviceToHost))) return -1;

			compareResults(params->ni, params->nj, params->BCPU, params->BGPU);
            // Store data if no warm up iteration
            if(rep>=0){
                printf("Elapsed time of kernel %d: %fms\n",kernel,milliseconds);
                params->kernelTimes[params->nof_repetitions*kernel+rep]=milliseconds;

                // Copyback times
                if (CheckCUDAError(cudaMemcpy(&params->blockTimes[2*params->nofBlocks*params->nof_repetitions*kernel + 2*params->nofBlocks*rep], \
                                kernelData[kernel].targetTimes, \
                                2*params->nofBlocks*sizeof(uint64_t), \
                                cudaMemcpyDeviceToHost))) return -1;

                // Copyback smid's
                if (CheckCUDAError(cudaMemcpy(&params->smid[params->nofBlocks*params->nof_repetitions*kernel + params->nofBlocks*rep], \
                                kernelData[kernel].smid, \
                                params->nofBlocks*sizeof(unsigned int), \
                                cudaMemcpyDeviceToHost))) return -1;
            }
        }
    }

    return 0;
}

int writeResults(param_t *params){
    if (fprintf(params->fd,"{\n") < 0 ) return -1;
    // Write device info
    cudaDeviceProp deviceProp;
    if (CheckCUDAError(cudaGetDeviceProperties(&deviceProp, DEVICE_NUMBER))) return -1;
    if (fprintf(params->fd,"\"clockRate\": \"%d\",\n", deviceProp.clockRate)  < 0 ) return -1;

    // Write header
    if (fprintf(params->fd,"\"nofThreads\": \"%u\",\n", params->nofThreads)  < 0 ) return -1;
    if (fprintf(params->fd,"\"nofBlocks\": \"%u\",\n", params->nofBlocks)  < 0 ) return -1;
    if (fprintf(params->fd,"\"nofKernel\": \"%u\",\n", params->nofKernels)  < 0 ) return -1;
    if (fprintf(params->fd,"\"nof_repetitions\": \"%d\",\n", params->nof_repetitions)  < 0 ) return -1;
    if (fprintf(params->fd,"\"data_size\": \"%d\",\n", params->ni*params->nj)  < 0 ) return -1;
    if (fprintf(params->fd,"\"measOH\": \"%lu\",\n", params->hostMeasOH)  < 0 ) return -1;

    // Write times
    int size_time = params->nofKernels * 2*params->nofBlocks * params->nof_repetitions;

    if (fprintf(params->fd,"\"blocktimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%lu\",\n",params->blockTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lu\"],\n", params->blockTimes[size_time-1]) < 0 ) return -1;

    size_time = params->nofKernels * params->nof_repetitions;
    if (fprintf(params->fd,"\"kerneltimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%f\",\n",params->kernelTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%f\"],\n", params->kernelTimes[size_time-1]) < 0 ) return -1;

    size_time = params->nofKernels * params->nofBlocks*params->nof_repetitions;
    if (fprintf(params->fd,"\"smid\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%u\",\n",params->smid[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%u\"]\n}", params->smid[size_time-1]) < 0 ) return -1;

    if (fclose(params->fd) < 0) return -1;
    return 0;
}

int cleanUp(param_t *params){
    kernel_data_t *kernelData = (kernel_data_t*)params->kernelData;

    for(int kernel = 0; kernel < params->nofKernels; kernel++){
        cudaEventDestroy(kernelData[kernel].start);
        cudaEventDestroy(kernelData[kernel].stop);
        cudaStreamDestroy(kernelData[kernel].stream);
        if(kernelData->A != NULL) cudaFree(kernelData->A);
        if(kernelData->B != NULL) cudaFree(kernelData->B);
        if(kernelData->targetTimes != NULL) cudaFree(kernelData->targetTimes);
        if(kernelData->smid != NULL) cudaFree(kernelData->smid);
    }

    // Free target buffers
    if(params->targetMeasOH != NULL) cudaFree(params->targetMeasOH);

    // Free host buffers
    if(params->kernelTimes != NULL) free(params->kernelTimes);
    if(params->blockTimes != NULL) free(params->blockTimes);
    if(params->BCPU != NULL) free(params->BCPU);
    if(params->BGPU != NULL) free(params->BGPU);
    if(params->smid != NULL) free(params->smid);
    if(params->kernelData != NULL) free(params->kernelData);

    cudaDeviceReset();
    return 0;
}
