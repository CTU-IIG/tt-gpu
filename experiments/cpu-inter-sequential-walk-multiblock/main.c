#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <err.h>
#include <sched.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdlib.h>
#include <error.h>
#include "interference.h"
#include "sequential-walk.h"
#include "sem.h"


#define MAX_CORES		6

#define CHECK(sts,msg)  \
	if (sts == -1) {      \
		perror(msg);        \
		exit(-1);           \
	}

typedef enum{
    INTER_RAND,
    INTER_SEQ,
    INTER_NO
} inter_t;

static pthread_barrier_t barrier;
static volatile uint32_t finishInference = 0;
static volatile uint32_t startInf = 0;
static int interferenceMethod = 0;

typedef struct{
    int cpu;
    param_t *params;
} threadData_t;

void compute_kernel(uint64_t time_us)
{
	struct timespec ts;
	uint64_t current_us, end_us;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	end_us = ts.tv_sec * 1000000000 + ts.tv_nsec  + time_us*1000;
	do {
		clock_gettime(CLOCK_MONOTONIC, &ts);
		current_us = ts.tv_sec * 1000000000 + ts.tv_nsec;
	} while (current_us < end_us);
}

void *cpu_thread(void *ptr)
{
	struct timespec ts;
	double start, end;
    int sum;
	cpu_set_t set;
	threadData_t *data = (threadData_t *)ptr;

	/* Ensure that our test thread does not migrate to another CPU
	 * during memguarding */
	CPU_ZERO(&set);
	CPU_SET(data->cpu, &set);
	if (sched_setaffinity(0, sizeof(set), &set) < 0)
		err(1, "sched_setaffinity");
    printf("Mem OP: %d\n", interferenceMethod);
    
    pthread_barrier_wait(&barrier);



    // Memory inference thread START
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = ts.tv_sec + ts.tv_nsec/1000000000;

    if(interferenceMethod == INTER_RAND){
        sum = interference_random(&finishInference, &startInf);
    } else if(interferenceMethod == INTER_SEQ){
        sum = interference_seq(&finishInference,&startInf);
    } else{
        // No interference
        sempost(&startInf);
        while(!finishInference);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts);
    end = ts.tv_sec + ts.tv_nsec/1000000000;
    // Memory inference thread END

    printf("cpu %d: interferenceTime: %lfs sum: %d\n", sched_getcpu(), end-start, sum);

    return NULL;
}




void *gpu_thread(void *ptr){
	cpu_set_t set;
	threadData_t *data = (threadData_t *)ptr;
    param_t *params = data->params;

	/* Ensure that our test thread does not migrate to another CPU
	 * during memguarding */
	CPU_ZERO(&set);
	CPU_SET(data->cpu, &set);

	//	struct sched_param param;
	//	int sts;
	//	int high_priority = sched_get_priority_max(SCHED_FIFO);
	//	//int high_priority = sched_get_priority_max(SCHED_OTHER);
	//	CHECK(high_priority,"sched_get_priority_max");
	//
	//	sts = sched_getparam(0, &param);
	//	CHECK(sts,"sched_getparam");
	//
	//	param.sched_priority = high_priority;
	//	sts = sched_setscheduler(0, SCHED_FIFO, &param);
	//	//sts = sched_setscheduler(0, SCHED_OTHER, &param);
	//	CHECK(sts,"sched_setscheduler");

	/* Ensure that our test thread does not migrate to another CPU
	 * during memguarding */

    finishInference=0;

	if (sched_setaffinity(0, sizeof(set), &set) < 0)
		err(1, "sched_setaffinity");

    pthread_barrier_wait(&barrier);

    //for(int i = 0; i<5;i++){
    //    semwait(&startInf);
    //}
    //compute_kernel(10000000);
#if 1
    // Initialize parameters
    if (initializeTest(params) < 0) return NULL;
	

    // Run test
    if (runTest(params) < 0) return NULL;

    // Write results
    if (writeResults(params) < 0){
        perror("Error while writing outpufile: ");
        return NULL;
    }

    // Clean up
    if (cleanUp(params) < 0) return NULL;
#endif
    finishInference=1;
	return NULL;
}


static int initThreads(param_t *params)
{
    finishInference=0;
	pthread_t threads[MAX_CORES];

	int s = pthread_barrier_init(&barrier, NULL, 1);//MAX_CORES);
	if (s != 0)
		error(1, s, "pthread_barrier_init");
    
    threadData_t data[MAX_CORES];
    data[0].cpu = 0;
    data[1].cpu = 1;
    data[2].cpu = 2;
    data[3].cpu = 3;
    data[4].cpu = 4;
    data[5].cpu = 5;
    data[5].params = params;

//	pthread_create(&threads[0], NULL, cpu_thread, (void *)&data[0]);
//	pthread_create(&threads[1], NULL, cpu_thread, (void *)&data[1]);
//	pthread_create(&threads[2], NULL, cpu_thread, (void *)&data[2]);
//	pthread_create(&threads[3], NULL, cpu_thread, (void *)&data[3]);
//	pthread_create(&threads[4], NULL, cpu_thread, (void *)&data[4]);
	pthread_create(&threads[5], NULL, gpu_thread, (void *)&data[5]);

//	pthread_join(threads[0], NULL);
//	pthread_join(threads[1], NULL);
//	pthread_join(threads[2], NULL);
//	pthread_join(threads[3], NULL);
//	pthread_join(threads[4], NULL);
	pthread_join(threads[5], NULL);

	return 0;
}


static void PrintUsage(const char *name) {
    printf("Usage: %s <#threads> <#blocks> <# kernel> <# of intervals> "
            "<size in KB> <interference method (0rnd or 1seq)"
            "<output JSON file name>\n", name);
}

int main(int argc, char **argv) {

    if (argc != 8) {
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
        printf("Min 1 kernel. Got %s blocks\n", argv[3]);
        return EXIT_FAILURE;
    }

    params.nofThreads = nof_threads;
    params.nofBlocks = nof_blocks;

    int nof_repetitions = atoi(argv[4]);
    if (nof_repetitions <= 0) {
        printf("More than 0 repetitions need to be used. Got %s repetitions\n", argv[4]);
        return EXIT_FAILURE;
    }

    int data_size = atoi(argv[5]);
    if (data_size <= 0) {
        printf("The buffer must be 1 or more KB. Got %s KB\n", argv[5]);
        return EXIT_FAILURE;
    }
    
    interferenceMethod = atoi(argv[6]);
    if (interferenceMethod < INTER_RAND || interferenceMethod > INTER_NO) {
        printf("Interference method must be 0,1 or 2. Got %s\n", argv[6]);
        return EXIT_FAILURE;
    }


    params.nof_repetitions = nof_repetitions;
    params.data_size = data_size*1024;
    params.buffer_length = data_size*1024/sizeof(int);

    params.fd = NULL;
    params.fd = fopen(argv[7],"w");
    if (params.fd == NULL) {
        perror("Error opening output file:");
        return EXIT_FAILURE;
    }

    initThreads(&params);
    printf("Finished testrun\n");
    return 0;
}
