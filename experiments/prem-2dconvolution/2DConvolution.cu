#include <err.h>
#include <sched.h>
#include <pthread.h>
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "2DConvolution.h"
#include "utility_func.h"

#define DEVICE_NUMBER (0)
#define START_TIME_OFFSET_NS (10000000) //10ms


#ifdef USE_PREM_PROF
#define TIME_PADDING (2500)
#endif

typedef struct {
    float *A;
    float *B;
    int ni;
    int nj;
    unsigned int kernelId;
    unsigned int nofKernel;
    uint64_t *targetTimes;
    uint64_t *startTime;
#ifdef USE_PREM_PROF
    unsigned int *tileCount;
    uint64_t *prefetchTimes;
    uint64_t *computeTimes;
    uint64_t *writebackTimes;
#endif
    unsigned int *smid;
    cudaStream_t stream;
    double *start;
    double *stop;
} kernel_data_t;


//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f


static float absVal(float a){
    if(a < 0){
        return (a * -1);
    }
    else{ 
        return a;
    }
}

static float percentDiff(double val1, double val2){
    if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)){
        return 0.0f;
    }

    else{
        return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
    }
}

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
static void compareResults(int ni, int nj, float *B, float *BGPU){
    int i, j, fail;
    fail = 0;

    // Compare outputs from CPU and GPU
    for (i=1; i < (ni-1); i++) {
        for (j=1; j < (nj-1); j++) {
            if (percentDiff(B[i*nj+j], BGPU[i*nj+j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
                printf("fail i: %d j: %d\n", i, j);
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}

static void conv2DCPU(int ni, int nj, float* A, float *B){
    int i, j;
    float c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
    c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
    c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


    for (i = 1; i < ni - 1; ++i) {
        for (j = 1; j < nj - 1; ++j){
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

static void initA(int ni, int nj, float* A){
    int i, j;

    for (i = 0; i < ni; ++i){
        for (j = 0; j < nj; ++j){
            A[nj*i+j] = (float)rand()/RAND_MAX;
        }
    }
}

static __global__ void getStartTime(param_t data) {
    if(threadIdx.x == 0){
        *data.targetStartTime = getTime() + START_TIME_OFFSET_NS;
    }
    __syncthreads();
}

// ----------------------------------- 
//         PREM SYNC functions
// ----------------------------------- 

static __device__ __inline__ void spinUntil(uint64_t endTime){
    if( threadIdx.x == 0){
        while(getTime() < endTime);
    }
}

#ifdef SCHEDULE
// Phase times manually evaluated
#if 0
#define PREM_PF_PHASE (2000)
#define PREM_C_PHASE (700)
#define PREM_WB_PHASE (500)
#else

// Phase times pessimistic
#define PREM_PF_PHASE (4000)
#define PREM_C_PHASE (2000)
#define PREM_WB_PHASE (2000)
#endif


/* Single phase schedule*/
#ifndef WB_SHIFT_BACK
#define WB_SHIFT_BACK (0)
#endif

#ifndef PREM_PF_PHASE_OFFSET
#define PREM_PF_PHASE_OFFSET (0)
#endif


#define WARM_UP_OFFSET (10000)
static __device__ __inline__ void syncPrefetch(const uint64_t startTime, const unsigned int tileId, const unsigned int id , const unsigned int tileoffset, const unsigned int phaseoffset){
    if(tileId > 0){
        if(threadIdx.x == 0){
                uint64_t delay = (startTime + WARM_UP_OFFSET) + ((tileId-1) * tileoffset) + (id * phaseoffset);
                while(getTime() < delay);
        }
    }
}

#ifdef PREM_SCHEDULE_ALL_PHASES

#ifndef PREM_WB_PHASE_OFFSET
#define PREM_WB_PHASE_OFFSET (0)
#endif

static __device__ __inline__ void syncWriteBack(const uint64_t startTime, const unsigned int tileId, const unsigned int id, const unsigned int pfTileOffset, const unsigned int wbTileOffset, unsigned int wbPhaseOffset){
    if(tileId > 0){
        if(threadIdx.x == 0){
            uint64_t delay = (startTime + WARM_UP_OFFSET) + ((tileId-1) * pfTileOffset) + wbTileOffset + (id * wbPhaseOffset);
            while(getTime() < delay);

        }
    }
}

#endif /*PREM_SCHEDULE_ALL_PHASES*/

#endif /*SCHEDULE*/

//#define PREM_SHM_SIZE (32768/(2*sizeof(float))) // Each kernel has 32kBytes of SHM ni=4 nj=1024 inputdata: 4096x4090 
#define PREM_SHM_SIZE (32768/(4*sizeof(float))) // Each kernel has 32kBytes of SHM ni=4, nj=512 inputdata: 4098 4082, 1026x1022
#define PREM_NJ_OVERLAP (2)

#define PREM_NI_TILE_SIZE (4)
#define PREM_NJ_TILE_SIZE (PREM_SHM_SIZE/PREM_NI_TILE_SIZE)

// Premification for datasets of size 512x512, 1024x1024 and 4096x4090
static __global__ void convolution2D_kernelPREM(kernel_data_t data){
    __shared__ float A_SHM[PREM_SHM_SIZE];
    __shared__ float B_SHM[PREM_SHM_SIZE];

#ifdef USE_PREM_PROF
    uint64_t reg_prof_pf_start  = 0;
    uint64_t reg_prof_pf_end  = 0;
    uint64_t reg_prof_c_end  = 0;
    uint64_t reg_prof_wb_start  = 0;
    uint64_t reg_prof_wb_end  = 0;
#endif /*USE_PREM_PROF*/

    unsigned int reg_tileCount = 0;

#ifdef SCHEDULE

#ifdef KERNELWISE_SYNC
    unsigned int reg_blockId = data.kernelId;
#else
    unsigned int reg_blockId = (data.kernelId * gridDim.x) + blockIdx.x;
#endif /*KERNELWISE_SYNC*/

#ifndef PREM_SCHEDULE_ALL_PHASES

#ifdef KERNELWISE_SYNC
    //unsigned int reg_pftileoffset = ((data.nofKernel-1) * PREM_PF_PHASE_OFFSET) + (PREM_PF_PHASE + PREM_C_PHASE + PREM_WB_PHASE);
    unsigned int reg_pftileoffset = ((data.nofKernel) * PREM_PF_PHASE_OFFSET);
#else
    //unsigned int reg_pftileoffset = (((data.nofKernel * gridDim.x)-1) * PREM_PF_PHASE_OFFSET) + (PREM_PF_PHASE + PREM_C_PHASE + PREM_WB_PHASE);
    unsigned int reg_pftileoffset = (((data.nofKernel * gridDim.x)) * PREM_PF_PHASE_OFFSET);
#endif /*KERNELWISE_SYNC*/

#else /*PREM_SCHEDULE_ALL_PHASES*/

#ifdef KERNELWISE_SYNC
    unsigned int reg_wbtileoffset = ((data.nofKernel-1) * PREM_PF_PHASE_OFFSET) + PREM_PF_PHASE + PREM_C_PHASE;
    unsigned int reg_pftileoffset = reg_wbtileoffset + (((data.nofKernel-1) * PREM_WB_PHASE_OFFSET) + PREM_WB_PHASE);
#else
    unsigned int reg_wbtileoffset = (((data.nofKernel * gridDim.x)-1) * PREM_PF_PHASE_OFFSET) + PREM_PF_PHASE + PREM_C_PHASE;
    unsigned int reg_pftileoffset = reg_wbtileoffset + ((((data.nofKernel * gridDim.x)-1) * PREM_WB_PHASE_OFFSET) + PREM_WB_PHASE);
#endif /*KERNELWISE_SYNC*/

#endif /*PREM_SCHEDULE_ALL_PHASES*/

#endif /*SCHEDULE*/

    uint64_t reg_startTime = *data.startTime;
   
    // Spin until PREM schedule start time 
    spinUntil(reg_startTime);

    uint64_t block_start_time = getTime();
    if(threadIdx.x == 0){
        data.targetTimes[blockIdx.x*2] = block_start_time;
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


    for(int niTile = blockIdx.y; niTile <= ni-PREM_NI_TILE_SIZE; niTile += (PREM_NI_TILE_SIZE/2)){
        for(int njTile = blockIdx.x*(PREM_NJ_TILE_SIZE-PREM_NJ_OVERLAP); njTile <= nj-PREM_NJ_TILE_SIZE; njTile += (PREM_NJ_TILE_SIZE-PREM_NJ_OVERLAP)*gridDim.x){

            /* ---------------------------------- */
            //Prefetch data
            /* ---------------------------------- */
#ifdef SCHEDULE
            syncPrefetch(reg_startTime, reg_tileCount, reg_blockId, reg_pftileoffset, PREM_PF_PHASE_OFFSET);
            __syncthreads();
#endif /*SCHEDULE*/

#ifdef USE_PREM_PROF
            if(threadIdx.x == 0){
                reg_prof_pf_start = getTime(); 
            }
#endif /*USE_PREM_PROF*/
            for (int i = threadIdx.y; (i < PREM_NI_TILE_SIZE) && (i + niTile < ni); i += blockDim.y){
                for (int j = threadIdx.x; (j < PREM_NJ_TILE_SIZE) && (j + njTile < nj); j += blockDim.x) {
                    A_SHM[i * PREM_NJ_TILE_SIZE + j] = A[(i+niTile)*nj + (j+njTile)];
                }
            }
            __syncthreads();

#ifdef USE_PREM_PROF
            if(threadIdx.x == 0){
                reg_prof_pf_end = getTime();
            }
#endif /*USE_PREM_PROF*/
            /* ---------------------------------- */
            // Compute on SHM
            /* ---------------------------------- */
            for (int i = threadIdx.y+1; i < PREM_NI_TILE_SIZE-1; i += blockDim.y){
                for (int j = threadIdx.x+1; j < PREM_NJ_TILE_SIZE-1; j += blockDim.x) {
                    B_SHM[i * PREM_NJ_TILE_SIZE + j] = c11 * A_SHM[(i - 1) * PREM_NJ_TILE_SIZE + (j - 1)] + 
                        c21 * A_SHM[(i - 1) * PREM_NJ_TILE_SIZE + (j + 0)] + 
                        c31 * A_SHM[(i - 1) * PREM_NJ_TILE_SIZE + (j + 1)] + 
                        c12 * A_SHM[(i + 0) * PREM_NJ_TILE_SIZE + (j - 1)] + 
                        c22 * A_SHM[(i + 0) * PREM_NJ_TILE_SIZE + (j + 0)] + 
                        c32 * A_SHM[(i + 0) * PREM_NJ_TILE_SIZE + (j + 1)] + 
                        c13 * A_SHM[(i + 1) * PREM_NJ_TILE_SIZE + (j - 1)] + 
                        c23 * A_SHM[(i + 1) * PREM_NJ_TILE_SIZE + (j + 0)] + 
                        c33 * A_SHM[(i + 1) * PREM_NJ_TILE_SIZE + (j + 1)];

                }
            }

            __syncthreads();

#ifdef USE_PREM_PROF
            if(threadIdx.x == 0){
                reg_prof_c_end = getTime();
            }
#endif /*USE_PREM_PROF*/

#ifdef SCHEDULE
#ifdef PREM_SCHEDULE_ALL_PHASES
            syncWriteBack(reg_startTime, reg_tileCount, reg_blockId, reg_pftileoffset, reg_wbtileoffset-WB_SHIFT_BACK, PREM_WB_PHASE_OFFSET);
            __syncthreads();
#endif /*PREM_SCHEDULE_ONE_PHASE*/
#endif /*SCHEDULE*/

#ifdef USE_PREM_PROF
            if(threadIdx.x == 0){
#ifdef PREM_SCHEDULE_ALL_PHASES
                reg_prof_wb_start = getTime();
#else
                reg_prof_wb_start = reg_prof_c_end;
#endif /*PREM_SCHEDULE_ONE_PHASE*/
            }
#endif /*USE_PREM_PROF*/

            /* ---------------------------------- */
            // Write back data
            /* ---------------------------------- */
            for (int i = threadIdx.y+1; (i < PREM_NI_TILE_SIZE-1) && (i + niTile < ni-1); i += blockDim.y){
                for (int j = threadIdx.x+1; (j < PREM_NJ_TILE_SIZE-1) && (j + njTile < nj-1); j += blockDim.x) {
                    B[(i+niTile) * nj + (j+njTile)] = B_SHM[i*PREM_NJ_TILE_SIZE + j];

                }
            }

#ifdef USE_PREM_PROF
            if(threadIdx.x == 0){
                reg_prof_wb_end = getTime(); 
                // Write times to global memory
                data.prefetchTimes[blockIdx.x*TIME_PADDING+2*reg_tileCount] = reg_prof_pf_start;
                data.prefetchTimes[blockIdx.x*TIME_PADDING+2*reg_tileCount+1] = reg_prof_pf_end;
                data.computeTimes[blockIdx.x*TIME_PADDING+2*reg_tileCount] = reg_prof_pf_end;
                data.computeTimes[blockIdx.x*TIME_PADDING+2*reg_tileCount+1] = reg_prof_c_end;
                data.writebackTimes[blockIdx.x*TIME_PADDING+2*reg_tileCount] = reg_prof_wb_start;
                data.writebackTimes[blockIdx.x*TIME_PADDING+2*reg_tileCount+1] = reg_prof_wb_end;
            }
#endif
            if(threadIdx.x == 0){
                reg_tileCount++;
            }

        }
    }

    if(threadIdx.x == 0){
        data.targetTimes[blockIdx.x*2+1] = getTime();
#ifdef USE_PREM_PROF
        *data.tileCount = reg_tileCount;
#endif
    }
    /*
       if(threadIdx.x == 0 && blockIdx.x == 0){
       printf("Kernel %d, masterGT: %lu\n",kernelId, *data.masterGT);
       }
     */
}


static __global__ void convolution2D_kernelLegacy(kernel_data_t data){
    uint64_t reg_startTime = *data.startTime;
   
    // Spin until PREM schedule start time 
    spinUntil(reg_startTime);

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

#ifdef USE_PREM_PROF
    params->prefetchTimes=NULL;
    params->prefetchTimes = (uint64_t *) malloc(params->nofKernels * params->nofBlocks * TIME_PADDING *sizeof(uint64_t));
    if (!params->prefetchTimes) {
        perror("Failed allocating prefetch times on host: ");
        return  -1;
    }
    params->computeTimes=NULL;
    params->computeTimes = (uint64_t *) malloc(params->nofKernels * params->nofBlocks * TIME_PADDING *sizeof(uint64_t));
    if (!params->computeTimes) {
        perror("Failed allocating prefetch times on host: ");
        return  -1;
    }
    params->writebackTimes=NULL;
    params->writebackTimes = (uint64_t *) malloc(params->nofKernels * params->nofBlocks * TIME_PADDING *sizeof(uint64_t));
    if (!params->writebackTimes) {
        perror("Failed allocating prefetch times on host: ");
        return  -1;
    }
#endif

    // Allocate device startTime
    if (CheckCUDAError(cudaMalloc(&params->targetStartTime, \
                    sizeof(uint64_t)))) return -1;

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

        // Allocate start stop times
        kernelData[i].start=NULL;
        kernelData[i].start = (double *) malloc(sizeof(double));
        if (!kernelData[i].start) {
            perror("Failed allocating kernel start times on host: ");
            return  -1;
        }
        kernelData[i].stop=NULL;
        kernelData[i].stop = (double *) malloc(sizeof(double));
        if (!kernelData[i].stop) {
            perror("Failed allocating kernel stop times on host: ");
            return  -1;
        }

#ifdef USE_PREM_PROF
        // Allocate PREM profiling data
        if (CheckCUDAError(cudaMalloc(&kernelData[i].prefetchTimes , \
                        params->nofBlocks * TIME_PADDING *sizeof(uint64_t)))) return -1;    
        if (CheckCUDAError(cudaMalloc(&kernelData[i].computeTimes , \
                        params->nofBlocks * TIME_PADDING *sizeof(uint64_t)))) return -1;    
        if (CheckCUDAError(cudaMalloc(&kernelData[i].writebackTimes , \
                        params->nofBlocks * TIME_PADDING *sizeof(uint64_t)))) return -1;    
        if (CheckCUDAError(cudaMalloc(&kernelData[i].tileCount , \
                        sizeof(unsigned int)))) return -1;    
#endif
        kernelData[i].startTime = params->targetStartTime;
        kernelData[i].kernelId = i;
        kernelData[i].nofKernel = params->nofKernels;
        kernelData[i].ni = params->ni;
        kernelData[i].nj = params->nj;
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
    params->kernelTimes = (double *) malloc(params->nofKernels*params->nof_repetitions*sizeof(double));
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



    // Copyback target startTime
    params->hostStartTime=0; 
    if (CheckCUDAError(cudaMemcpy(&params->hostStartTime, \
                    params->targetStartTime, \
                    sizeof(uint64_t), \
                    cudaMemcpyDeviceToHost))) return -1;

    // Assigne kernel data
    params->kernelData = (void*) kernelData;

    if(A != NULL) free(A);

    return 0;
}


typedef struct{
    int cpu;
    int nofBlocks;
    int nofThreads;
    int usePrem;
    kernel_data_t kernelData;
} thread_data_t;

static pthread_barrier_t barrier;
static void *executeKernel(void * ptr){
    thread_data_t *threadData = (thread_data_t*)ptr;
    kernel_data_t kernelData = threadData->kernelData;
    int nofThreads = threadData->nofThreads;
    int nofBlocks = threadData->nofBlocks;  

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(threadData->cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) < 0)
        err(1, "sched_setaffinity");

    pthread_barrier_wait(&barrier);
    *threadData->kernelData.start = hostTimeMs();
    if(threadData->usePrem){
        convolution2D_kernelPREM<<<nofBlocks,\
            nofThreads,\
            0,\
            kernelData.stream>>>(kernelData);
    }else{
        convolution2D_kernelLegacy<<<nofBlocks,\
            nofThreads,\
            0,\
            kernelData.stream>>>(kernelData);

    }
    pthread_barrier_wait(&barrier);
    if (CheckCUDAError(cudaStreamSynchronize(kernelData.stream))) perror("Problem with stream sync");
    *threadData->kernelData.stop = hostTimeMs();

    return NULL;
}

int runTest(param_t *params){

    kernel_data_t *kernelData = (kernel_data_t*)params->kernelData;

    thread_data_t threadData[params->nofKernels];

    pthread_t threads[params->nofKernels];

    int s = pthread_barrier_init(&barrier, NULL, params->nofKernels);
    if (s != 0)
        error(1, s, "pthread_barrier_init");

    for(int rep = -1; rep < params->nof_repetitions; rep++){

        // Get measurement startTime
        getStartTime<<<1,1>>>(*params);

        if (CheckCUDAError(cudaDeviceSynchronize())) return -1;
        for(int kernel = 0; kernel < params->nofKernels; kernel++){
            threadData[kernel].cpu=kernel;
            threadData[kernel].nofThreads=params->nofThreads;
            threadData[kernel].nofBlocks=params->nofBlocks;
            threadData[kernel].kernelData = kernelData[kernel];
            threadData[kernel].usePrem = params->usePREM;
            pthread_create(&threads[kernel], NULL, executeKernel, (void *)&threadData[kernel]);
        }

        for(int kernel = 0; kernel < params->nofKernels; kernel++){
            pthread_join(threads[kernel], NULL);
        }

        if (CheckCUDAError(cudaDeviceSynchronize())) perror("Problem with stream sync");

        for(int kernel = 0; kernel < params->nofKernels; kernel++){
            double milliseconds = 0.0;
            if(params->usePREM){
                milliseconds = *kernelData[kernel].stop-*kernelData[kernel].start-(START_TIME_OFFSET_NS/1000000.0);
            } else {
                milliseconds = *kernelData[kernel].stop-*kernelData[kernel].start;
            }

            // Copyback B and check

            if (CheckCUDAError(cudaMemcpy(params->BGPU, \
                            kernelData[kernel].B, \
                            params->ni*params->nj*sizeof(float), \
                            cudaMemcpyDeviceToHost))) return -1;

            compareResults(params->ni, params->nj, params->BCPU, params->BGPU);
            // Store data if no warm up iteration
            if(rep>=0){
                printf("Elapsed time of kernel %d: %lfms\n",kernel,milliseconds);
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
#ifdef USE_PREM_PROF
                if (CheckCUDAError(cudaMemcpy(&params->tileCount, \
                                kernelData[kernel].tileCount, \
                                sizeof(unsigned int), \
                                cudaMemcpyDeviceToHost))) return -1;
                if (CheckCUDAError(cudaMemcpy(&params->prefetchTimes[params->nofBlocks*TIME_PADDING*kernel], \
                                kernelData[kernel].prefetchTimes, \
                                params->nofBlocks*TIME_PADDING*sizeof(uint64_t), \
                                cudaMemcpyDeviceToHost))) return -1;
                if (CheckCUDAError(cudaMemcpy(&params->computeTimes[params->nofBlocks*TIME_PADDING*kernel], \
                                kernelData[kernel].computeTimes, \
                                params->nofBlocks*TIME_PADDING*sizeof(uint64_t), \
                                cudaMemcpyDeviceToHost))) return -1;
                if (CheckCUDAError(cudaMemcpy(&params->writebackTimes[params->nofBlocks*TIME_PADDING*kernel], \
                                kernelData[kernel].writebackTimes, \
                                params->nofBlocks*TIME_PADDING*sizeof(uint64_t), \
                                cudaMemcpyDeviceToHost))) return -1;
#endif
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
    if (fprintf(params->fd,"\"start_time\": \"%lu\",\n", params->hostStartTime)  < 0 ) return -1;

    // Write times
#ifdef USE_PREM_PROF
    if (fprintf(params->fd,"\"tileCount\": \"%u\",\n", params->tileCount)  < 0 ) return -1;

    if (fprintf(params->fd,"\"prefetchtimes\":[\n") < 0 ) return -1;
    for(int j = 0; j< params->nofKernels*params->nofBlocks; j++){
        for (int i = j*TIME_PADDING; i < (j*TIME_PADDING)+2*params->tileCount; i++){
            if (fprintf(params->fd,"\"%lu\",\n",params->prefetchTimes[i]) < 0 ) return -1;
        }
    }
    if (fprintf(params->fd,"\"%lu\"],\n", params->prefetchTimes[1]) < 0 ) return -1;

    if (fprintf(params->fd,"\"computetimes\":[\n") < 0 ) return -1;
    for(int j = 0; j< params->nofKernels*params->nofBlocks; j++){
        for (int i = j*TIME_PADDING; i < (j*TIME_PADDING)+2*params->tileCount; i++){
            if (fprintf(params->fd,"\"%lu\",\n",params->computeTimes[i]) < 0 ) return -1;
        }
    }
    if (fprintf(params->fd,"\"%lu\"],\n", params->computeTimes[1]) < 0 ) return -1;

    if (fprintf(params->fd,"\"writebacktimes\":[\n") < 0 ) return -1;
    for(int j = 0; j< params->nofKernels*params->nofBlocks; j++){
        for (int i = j*TIME_PADDING; i < (j*TIME_PADDING)+2*params->tileCount; i++){
            if (fprintf(params->fd,"\"%lu\",\n",params->writebackTimes[i]) < 0 ) return -1;
        }
    }
    if (fprintf(params->fd,"\"%lu\"],\n", params->writebackTimes[1]) < 0 ) return -1;
#endif
    int size_time = params->nofKernels * 2*params->nofBlocks * params->nof_repetitions;

    if (fprintf(params->fd,"\"blocktimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%lu\",\n",params->blockTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lu\"],\n", params->blockTimes[size_time-1]) < 0 ) return -1;

    size_time = params->nofKernels * params->nof_repetitions;
    if (fprintf(params->fd,"\"kerneltimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%lf\",\n",params->kernelTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lf\"],\n", params->kernelTimes[size_time-1]) < 0 ) return -1;

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
        cudaStreamDestroy(kernelData[kernel].stream);
        if(kernelData->A != NULL) cudaFree(kernelData->A);
        if(kernelData->B != NULL) cudaFree(kernelData->B);
        if(kernelData->targetTimes != NULL) cudaFree(kernelData->targetTimes);
        if(kernelData->smid != NULL) cudaFree(kernelData->smid);
#ifdef USE_PREM_PROF
        if(kernelData->tileCount != NULL) cudaFree(kernelData->tileCount);
        if(kernelData->prefetchTimes != NULL) cudaFree(kernelData->prefetchTimes);
        if(kernelData->computeTimes != NULL) cudaFree(kernelData->computeTimes);
        if(kernelData->writebackTimes != NULL) cudaFree(kernelData->writebackTimes);
#endif
    }

    // Free target buffers
    if(params->targetStartTime != NULL) cudaFree(params->targetStartTime);

    // Free host buffers
#ifdef USE_PREM_PROF
    if(params->prefetchTimes != NULL)  free(params->prefetchTimes);
    if(params->computeTimes != NULL)   free(params->computeTimes);
    if(params->writebackTimes != NULL) free(params->writebackTimes);
#endif
    if(params->kernelTimes != NULL) free(params->kernelTimes);
    if(params->blockTimes != NULL) free(params->blockTimes);
    if(params->BCPU != NULL) free(params->BCPU);
    if(params->BGPU != NULL) free(params->BGPU);
    if(params->smid != NULL) free(params->smid);
    if(params->kernelData != NULL) free(params->kernelData);

    cudaDeviceReset();
    return 0;
}
