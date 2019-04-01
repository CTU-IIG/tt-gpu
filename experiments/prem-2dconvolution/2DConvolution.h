#ifndef CONV2D_H
#define CONV2D_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nofThreads;
    int nofBlocks;
    int nofKernels;
    int32_t nof_repetitions;
    int ni;
    int nj;
    float *BCPU;
    float *BGPU;
    float *kernelTimes;
    uint64_t *blockTimes;
    unsigned int *smid;
    uint64_t *targetMeasOH;
    uint64_t hostMeasOH;
    void * kernelData;
    FILE *fd;
} param_t;

int initializeTest(param_t *params);

int runTest(param_t *params);

int writeResults(param_t *params);

int cleanUp(param_t *params);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
