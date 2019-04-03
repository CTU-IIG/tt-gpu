#ifndef CONV2D_H
#define CONV2D_H
#ifdef __cplusplus
extern "C" {
#endif

//#define USE_PREM_PROF

typedef struct {
    int nofThreads;
    int nofBlocks;
    int nofKernels;
    int32_t nof_repetitions;
    int usePREM;
    int ni;
    int nj;
    float *BCPU;
    float *BGPU;
    double *kernelTimes;
    uint64_t *blockTimes;
    unsigned int *smid;
    uint64_t *targetMeasOH;
    uint64_t hostMeasOH;
    void * kernelData;
    FILE *fd;
#ifdef USE_PREM_PROF
    int tileCount;
    uint64_t *prefetchTimes;
    uint64_t *computeTimes;
    uint64_t *writebackTimes;
#endif
} param_t;

int initializeTest(param_t *params);

int runTest(param_t *params);

int writeResults(param_t *params);

int cleanUp(param_t *params);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
