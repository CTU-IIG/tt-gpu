#ifndef UTIL_FUNC_H
#define UTIL_FUNC_H
#ifdef __cplusplus
extern "C" {
#endif

// Prints a message and returns zero if the given value is not cudaSuccess
#define CheckCUDAError(val) (InternalCheckCUDAError((val), #val, __FILE__, __LINE__))

// Called internally by CheckCUDAError
static inline int InternalCheckCUDAError(cudaError_t result, const char *fn,
        const char *file, int line) {
    if (result == cudaSuccess) return 0;
    printf("CUDA error %d in %s, line %d (%s): %s\n", (int) result, file, line,
            fn, cudaGetErrorString(result));
    return -1;
}


static inline double hostTimeMs(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return (((double) ts.tv_sec)*1e3) + (((double) ts.tv_nsec) / 1e6);
}

static __device__ __inline__ uint64_t getTime(void){
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

static __device__ __inline__ unsigned int get_smid(void)
{
    unsigned int ret;
    asm("mov.u32 %0, %%smid;":"=r"(ret) );
    return ret;
}

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
