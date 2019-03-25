#ifndef SEQ_WALK_H
#define SEQ_WALK_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nofThreads;
    int nofBlocks;
    int nofKernel;
    int32_t nof_repetitions;
    int data_size;
    int buffer_length;
    unsigned int hostMeasOH;
    int *hostBuffer;
    uint64_t host_realSum;
    unsigned int *targetMeasOH;
    unsigned int *host_times;
    uint64_t *target_realSum;
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
