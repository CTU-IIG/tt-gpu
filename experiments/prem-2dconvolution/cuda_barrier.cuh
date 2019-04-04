#ifndef CUDA_BARRIER_H
#define CUDA_BARRIER_H
#ifdef __cplusplus
extern "C" {
#endif
// membar implementation simmilar to https://bigfoot.cs.unc.edu:3000/otternes/cuda_scheduling_examiner/src/master/src/barrier_wait.c
typedef struct {
    int *threadsRemaining;
    int *sense;
    int thread_count;
} barrier_t;

static __device__ inline int barrierWait(barrier_t barrier, int* local_sense) {
    *local_sense = !(*local_sense);
    int value = atomicSub(barrier.threadsRemaining, 1);

    if(value==1) {

    atomicExch(barrier.threadsRemaining, barrier.thread_count);
    *(barrier.sense) = *local_sense;
    return 1;
    }

    while (*(barrier.sense) != *local_sense){
        asm volatile("membar.gl;" : : :);
        continue;
    }
    return 1;
}

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
