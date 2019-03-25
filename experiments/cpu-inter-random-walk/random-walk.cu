#include <errno.h>
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "random-walk.h"

#define DEVICE_NUMBER (0)

typedef struct {
	int32_t nof_repetitions;
	size_t buffer_length;
	uint32_t *targetMeasOH;
	uint32_t hostMeasOH;
	uint32_t *hostBuffer;
	uint32_t *targetBuffer;
	uint64_t *target_realSum;
	uint64_t host_realSum;
	unsigned int *target_times;
	unsigned int *host_times;
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
	for (uint32_t i = 0; i<nofElem;i++){
		uint32_t rndi, tmp1, tmp2, tmp3; 
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

static __global__ void getMeasurementOverhead(kernel_param_t params) {
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


static __global__ void randomWalk(kernel_param_t params) {
	uint32_t current;
	unsigned int time_start, time_end;
    unsigned int time_acc;
	uint64_t sum;
	unsigned int oh = *params.targetMeasOH;

	if (blockIdx.x != 0) return;
	if (threadIdx.x != 0) return;
	// Warm up data cache    
	for(int i = 0; i < params.buffer_length; i++){
		sum += params.targetBuffer[i];
	}

	// Run experiment multiple times. First iteration (-1) is to warm up icache
	for (int i = -2; i < params.nof_repetitions; i++){
		sum = 0;
	    current = 0;
		time_start = clock();
		for(int j = 0; j < params.buffer_length; j++){
			current = params.targetBuffer[current];
			sum += current;
		}
		time_end = clock();
		*params.target_realSum = sum;

		time_acc = (time_end - time_start);

		// Do not write time for warm up iteration       
		if (i>=0){
			// Write element access time with measurement overhead
			params.target_times[i] = (time_acc/params.buffer_length)-oh;
		}
	}
}

int initializeTest(param_t *params){
	// Set CUDA device
	if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))) {
		return EXIT_FAILURE;
	}

    // Allocate kernel parameter holder
    kernel_param_t *kernelParams = NULL;
    kernelParams = (kernel_param_t *)malloc(sizeof(kernel_param_t)); 
    if(kernelParams == NULL){
		perror("Failed allocating kernelparameter holder: ");
		return  -1;
    }
    kernelParams->nof_repetitions = params->nof_repetitions;
    // Set buffer length
	kernelParams->buffer_length = params->data_size/sizeof(uint32_t);

    if(params->useZeroCopy){
        // Use host pinned zerocopy memory
        if (CheckCUDAError(cudaHostAlloc(&kernelParams->hostBuffer, kernelParams->buffer_length*sizeof(uint32_t), cudaHostAllocMapped))) return -1;    
        if (CheckCUDAError(cudaHostGetDevicePointer((void **)&kernelParams->targetBuffer, (void *)kernelParams->hostBuffer, 0))) return -1;    
        
        
    }else{
        // Dont use host pinned zerocopy memory
        //allocate buffer
        kernelParams->hostBuffer = NULL;
        kernelParams->hostBuffer = (uint32_t *) malloc(kernelParams->buffer_length*sizeof(uint32_t));
        if (!kernelParams->hostBuffer) {
            perror("Failed allocating host buffer: ");
            return  -1;
        }
        //allocate device random buffer
        if (CheckCUDAError(cudaMalloc(&kernelParams->targetBuffer, \
                        kernelParams->buffer_length*sizeof(uint32_t)))) return -1;    
    }


	if (createShuffledArray(kernelParams->hostBuffer, kernelParams->buffer_length) != 0) return EXIT_FAILURE;

    if(!params->useZeroCopy){
        if (CheckCUDAError(cudaMemcpy(kernelParams->targetBuffer, \
                        kernelParams->hostBuffer, \
                        kernelParams->buffer_length*sizeof(uint32_t), \
                        cudaMemcpyHostToDevice))) return -1;
    }

	//allocate device times
	if (CheckCUDAError(cudaMalloc(&kernelParams->target_times, \
					kernelParams->nof_repetitions*sizeof(unsigned int)))) return -1;

	// Allocate device accumulator
	if (CheckCUDAError(cudaMalloc(&kernelParams->target_realSum, \
					sizeof(uint64_t)))) return -1;

	// Allocate device measOH
	if (CheckCUDAError(cudaMalloc(&kernelParams->targetMeasOH, \
					sizeof(uint32_t)))) return -1;

	//allocate host times
	kernelParams->host_times = NULL;
	kernelParams->host_times = (unsigned int *) malloc(kernelParams->nof_repetitions*sizeof(unsigned int));
	if (!kernelParams->host_times) {
		perror("Failed allocating host_times buffer: ");
		return  -1;
	}
	memset(kernelParams->host_times,0, kernelParams->nof_repetitions*sizeof(unsigned int));

    // Store kernel param holder to param_t struct
    params->kernelParam = (void*)kernelParams;

	return 0;
}

int runTest(param_t *params){

    kernel_param_t * kernelParam = (kernel_param_t*) params->kernelParam;
	getMeasurementOverhead<<<1,1>>>(*kernelParam);
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

	randomWalk<<<1,1>>>(*kernelParam);
	// Synchronize with device
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

	// Copyback times
	if (CheckCUDAError(cudaMemcpy(kernelParam->host_times, \
					kernelParam->target_times, \
					kernelParam->nof_repetitions*sizeof(unsigned int), \
					cudaMemcpyDeviceToHost))) return -1;

	// Copyback sum
	kernelParam->host_realSum=0; 
	if (CheckCUDAError(cudaMemcpy(&kernelParam->host_realSum, \
					kernelParam->target_realSum, \
					sizeof(uint64_t), \
					cudaMemcpyDeviceToHost))) return -1;

	// Copyback target meas oh
	kernelParam->hostMeasOH=0; 
	if (CheckCUDAError(cudaMemcpy(&kernelParam->hostMeasOH, \
					kernelParam->targetMeasOH, \
					sizeof(uint32_t), \
					cudaMemcpyDeviceToHost))) return -1;
	return 0;
}

int writeResults(param_t *params){

    kernel_param_t * kernelParam = (kernel_param_t*) params->kernelParam;

	if (fprintf(params->fd,"{\n") < 0 ) return -1;

	// Write header
	if (fprintf(params->fd,"\"nofThreads\": \"%u\",\n", 1)  < 0 ) return -1;
	if (fprintf(params->fd,"\"nofBlocks\": \"%u\",\n", 1)  < 0 ) return -1;
	if (fprintf(params->fd,"\"nof_repetitions\": \"%d\",\n", params->nof_repetitions)  < 0 ) return -1;
	if (fprintf(params->fd,"\"data_size\": \"%d\",\n", params->data_size)  < 0 ) return -1;
	if (fprintf(params->fd,"\"buffer_length\": \"%zu\",\n", kernelParam->buffer_length)  < 0 ) return -1;
	if (fprintf(params->fd,"\"real_sum\": \"%llu\",\n", (unsigned long long)kernelParam->host_realSum)  < 0 ) return -1;
	if (fprintf(params->fd,"\"exp_sum\": \"%lu\",\n", ((kernelParam->buffer_length-1)*kernelParam->buffer_length)/2)  < 0 ) return -1;
	if (fprintf(params->fd,"\"measOH\": \"%u\",\n", kernelParam->hostMeasOH)  < 0 ) return -1;

	// Write times
	if (fprintf(params->fd,"\"times\":[\n") < 0 ) return -1;
	for (int32_t i = 0; i < params->nof_repetitions-1; i++){
		if (fprintf(params->fd,"\"%u\",\n",kernelParam->host_times[i]) < 0 ) return -1;
	}
	if (fprintf(params->fd,"\"%u\"]\n}", kernelParam->host_times[params->nof_repetitions-1]) < 0 ) return -1;

	if (fclose(params->fd) < 0) return -1;
	return 0;
}

int cleanUp(param_t *params){
    kernel_param_t * kernelParam = (kernel_param_t*) params->kernelParam;
	// Free target buffers
    if(!params->useZeroCopy){
        cudaFree(kernelParam->targetBuffer);
	    free(kernelParam->hostBuffer);
    }else{
        cudaFreeHost(kernelParam->hostBuffer);
    }
	cudaFree(kernelParam->target_times);
	cudaFree(kernelParam->target_realSum);
	cudaFree(kernelParam->targetMeasOH);

	// Free host buffers
	free(kernelParam->host_times);

    // Free kernel parameter holder
    free(kernelParam);
    
	cudaDeviceReset();
	return 0;
}
