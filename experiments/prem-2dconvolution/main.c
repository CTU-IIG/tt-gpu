#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <err.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "2DConvolution.h"

static int startBenchmark(param_t *params)
{

	cpu_set_t set;
	/* Ensure that our test thread does not migrate to another CPU
	 * during memguarding */
	CPU_ZERO(&set);
	CPU_SET(5, &set);

    if (initializeTest(params) < 0) return -1;
	

    // Run test
    if (runTest(params) < 0) return -1;

    // Write results
    if (writeResults(params) < 0){
        perror("Error while writing outpufile: ");
        return -1;
    }

    // Clean up
    if (cleanUp(params) < 0) return -1;

	return 0;
}


static void PrintUsage(const char *name) {
    printf("Usage: %s <#threads> <#blocks> <# kernel> <# of intervals> "
            "<ni> <nj> <usePREM 1/0>"
            "<output JSON file name>\n", name);
}

int main(int argc, char **argv) {

    if (argc != 9) {
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
    params.nofKernels = nof_kernel;

    int nof_repetitions = atoi(argv[4]);
    if (nof_repetitions <= 0) {
        printf("More than 0 repetitions need to be used. Got %s repetitions\n", argv[4]);
        return EXIT_FAILURE;
    }

    int ni = atoi(argv[5]);
    if (ni <= 0) {
        printf("ni must be bigger than. Got %s\n", argv[5]);
        return EXIT_FAILURE;
    }
    int nj = atoi(argv[6]);
    if (nj <= 0) {
        printf("nj must be bigger than 0. Got %s\n", argv[6]);
        return EXIT_FAILURE;
    }

    params.ni = ni;
    params.nj = nj;
    
    
    params.usePREM = atoi(argv[7]);
    if (params.usePREM < 0 || params.usePREM > 1) {
        printf("Specify if Premized or legacy kernel should be used (1 or 0). Got %s\n", argv[7]);
        return EXIT_FAILURE;
    }

    params.nof_repetitions = nof_repetitions;

    params.fd = NULL;
    params.fd = fopen(argv[8],"w");
    if (params.fd == NULL) {
        perror("Error opening output file:");
        return EXIT_FAILURE;
    }

    startBenchmark(&params);
    printf("Finished testrun\n");
    return 0;
}
