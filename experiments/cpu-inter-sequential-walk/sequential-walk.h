#ifndef SEQ_WALK_H
#define SEQ_WALK_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t nof_repetitions;
    int data_size;
    FILE *fd;
    void *kernelParam;
    int useZeroCopy;       //0= do not use, 1= use
} param_t;

int initializeTest(param_t *params);

int runTest(param_t *params);

int writeResults(param_t *params);

int cleanUp(param_t *params);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
