#include <stdio.h>
#include <time.h>
#include <stdint.h>
#define DSIZE 12

// Returns the value of CUDA's global nanosecond timer.
static __device__ inline uint64_t GlobalTimer64(void) {
    // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
    // around after the upper bits have already been read. Work around this by
    // reading the high bits a second time. Use the second value to detect a
    // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
    // would be valid, it's passed over during the duration of the reading. If no
    // rollover occurred, just return the initial reading.
    volatile uint64_t first_reading;
    volatile uint32_t second_reading;
    uint32_t high_bits_first;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
    high_bits_first = first_reading >> 32;
    asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
    if (high_bits_first == second_reading) {
        return first_reading;
    }
    // Return the value with the updated high bits, but the low bits set to 0.
    return ((uint64_t) second_reading) << 32;
}

static __device__ __inline__ uint32_t __myclock(){
    uint32_t mclk;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(mclk));
    return mclk;
}

    __global__ void kernel(unsigned int*data_clock, long long int* data_clock64){
            unsigned int temp1, temp2, temp3; 
            long long int temp4, temp5, temp6;

            // Warm up l1 data cache
            for ( int i = 0; i< DSIZE; i++){
                data_clock[i] = 0;
                data_clock64[i] = 0;
            }

            // Clock normal
            // Warm icache
            temp1 = (unsigned int)clock();
            
            temp1 = (unsigned int)clock(); //R3
            temp2 = (unsigned int)clock();
            data_clock[0] = temp1;
            temp3 = (unsigned int)clock();
            data_clock[1] = temp2;
            data_clock[2] = temp3;
            data_clock[3] = temp2-temp1;
            data_clock[4] = temp3-temp2;
            data_clock[5] = data_clock[4] - data_clock[3];

            // Clock inline
            // Warm up icache
            temp1 = (unsigned int)__myclock();

            temp1 = (unsigned int)__myclock();
            temp2 = (unsigned int)__myclock();
            data_clock[6] = temp1;
            temp3 = (unsigned int)__myclock();
            data_clock[7] = temp2;
            data_clock[8] = temp3;
            data_clock[9] = temp2-temp1;
            data_clock[10] = temp3-temp2;
            data_clock[11] = data_clock[10] - data_clock[9];

            // Clock64() normal
            // Warm up icache
            temp4 = clock64();

            temp4 = clock64();
            temp5 = clock64();
            data_clock64[0] = temp4;
            temp6 = clock64();
            data_clock64[1] = temp5;
            data_clock64[2] = temp6;
            data_clock64[3] = temp5-temp4;
            data_clock64[4] = temp6-temp5;
            data_clock64[5] = data_clock64[4] - data_clock64[3];
            
            // GlobalTimer64() normal
            // Warm up icache
            uint64_t temp7 = GlobalTimer64();

            temp7 = GlobalTimer64();
            uint64_t temp8 = GlobalTimer64();

            temp1 = clock();
            //dummy
            uint64_t temp9 = GlobalTimer64();
            temp2 = clock();

            data_clock64[6] = (long long int) temp7;
            data_clock64[7] = (long long int)temp8;
            data_clock64[8] = (long long int)(temp8-temp7);
            data_clock64[9] = (long long int)(temp2-temp1);
            data_clock64[10] = (long long int)(temp9);
    }

int main(){

    unsigned int hdata_clock[DSIZE];
    unsigned int* ddata_clock;
    cudaMalloc(&ddata_clock, sizeof(hdata_clock));

    long long int hdata_clock64[DSIZE];
    long long int* ddata_clock64;
    cudaMalloc(&ddata_clock64, sizeof(hdata_clock64));

    kernel<<<1,1>>>(ddata_clock, ddata_clock64);

    cudaMemcpy(hdata_clock, ddata_clock, sizeof(hdata_clock), cudaMemcpyDeviceToHost);
    cudaMemcpy(hdata_clock64, ddata_clock64, sizeof(hdata_clock64), cudaMemcpyDeviceToHost);

    printf("timing of clock() (unsigned int) by using:\n");
    printf("temp1 = clock();\n");
    printf("temp2 = clock();\n");
    printf("data[0] = temp1;                   // [%u] cycles\n", hdata_clock[0]);
    printf("temp3 = clock();\n");
    printf("data[1] = temp2;                   // [%u] cycles\n", hdata_clock[1]);
    printf("data[2] = temp3;                   // [%u] cycles\n", hdata_clock[2]);
    printf("data[3] = temp2-temp1;             // [%u] cycles\n", hdata_clock[3]);
    printf("data[4] = temp3-temp2;             // [%u] cycles\n", hdata_clock[4]);
    printf("Global element store time:            [%u] cycles\n", hdata_clock[5]);
    printf("=================================================\n");
    printf("timing of inlined __myclock() (unsigned int) by using:\n");
    printf("temp1 = __myclock();\n");
    printf("temp2 = __myclock();\n");
    printf("data[5] = temp1;                   // [%u] cycles\n", hdata_clock[6]);
    printf("temp3 = __myclock();\n");
    printf("data[6] = temp2;                   // [%u] cycles\n", hdata_clock[7]);
    printf("data[7] = temp3;                   // [%u] cycles\n", hdata_clock[8]);
    printf("data[8] = temp2-temp1;             // [%u] cycles\n", hdata_clock[9]);
    printf("data[9] = temp3-temp2;             // [%u] cycles\n", hdata_clock[10]);
    printf("Global element store time:            [%u] cycles\n", hdata_clock[11]);
    printf("=================================================\n");
    printf("timing of clock64() (long long int) by using:\n");
    printf("temp4 = clock64();\n");
    printf("temp5 = clock64();\n");
    printf("data_clock64 [0] = temp4;          // [%lld] cycles\n", hdata_clock64[0]);
    printf("temp6 = clock64();\n");
    printf("data_clock64[1] = temp5;           // [%lld] cycles\n", hdata_clock64[1]);
    printf("data_clock64[2] = temp6;           // [%lld] cycles\n", hdata_clock64[2]);
    printf("data_clock64[3] = temp5-temp4;     // [%lld] cycles\n", hdata_clock64[3]);
    printf("data_clock64[4] = temp6-temp5;     // [%lld] cycles\n", hdata_clock64[4]);
    printf("Global element store time             [%lld] cycles\n", hdata_clock64[5]);
    printf("=================================================\n");
    printf("timing of GlobalTimer64() (uint64_t) by using:\n");
    printf("temp4 = (long long int)GlobalTimer64();\n");
    printf("temp5 = (long long int)GlobalTimer64();\n");
    printf("data_clock64 [0] = temp4;          // [%llu] ns\n", hdata_clock64[6]);
    printf("data_clock64[1] = temp5;           // [%llu] ns\n", hdata_clock64[7]);
    printf("data_clock64[3] = temp5-temp4;     // [%llu] ns\n", hdata_clock64[8]);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    double nofcycles = (double)(hdata_clock64[8]/1000000000.0)*(deviceProp.clockRate*1000.0);
    printf("Clock rate: %dkHz\n", deviceProp.clockRate);
    printf("data_clock64[3] = temp5-temp4;     // [%lf] cycles\n", nofcycles);
    printf("Measured [%llu] cycles\n", hdata_clock64[9]);
    return 0;
}
