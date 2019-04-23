#include <stdlib.h> 
#include <stdint.h>
#include "sem.h"
#include "interference.h"
#include <string.h>
#include <time.h>
#include <errno.h>
#include <stdio.h>

typedef struct{
    int index; //4 byte
    char dummy[58]; // 58 byte
} elem_t;

#define ARRAY_SIZE ((8*1024*1024)/sizeof(elem_t)) // 4MB of array
#define CACHE_LINE_SIZE (64)///sizeof(int))

/*! \brief  Create random 32bit number
 *  \return returns random 32bit number
 *  Uses rand() function to create a random 32 bit number using two calls
 */
__attribute__((always_inline)) inline uint32_t random32(void){
	//return (rand() ^ (rand() << 15));
    return (uint32_t)lrand48();
}

int interference_random(volatile uint32_t *finishInference, volatile uint32_t *startInf){
    elem_t *src;
    elem_t *dst;

    src = NULL;
    dst = NULL;

    src = (elem_t*)malloc(ARRAY_SIZE*sizeof(elem_t));
    dst = (elem_t*)malloc(ARRAY_SIZE*sizeof(elem_t));

    if(dst == NULL || src == NULL){
        perror("Error: Could not allocate buffers");
        return -1;
    }

    int sum = 0;

    printf("Init Rand array Elemsize: %lu\n", sizeof(elem_t));

	// Seed random
	//srand(time(NULL));
    srand48(time(NULL));

	// Link sequentially
	for(uint32_t i = 0; i< ARRAY_SIZE-1; i++){
		src[i].index = i+1;
        dst[i].index = i;
	}
	src[ARRAY_SIZE-1].index = 0;
	dst[ARRAY_SIZE-1].index = ARRAY_SIZE-1;

	// Shuffle array	
	for (uint32_t i = 0; i<ARRAY_SIZE;i++){
		uint32_t rndi, tmp1, tmp2, tmp3; 
		rndi = random32()%ARRAY_SIZE;
		if (rndi == i) continue;	

		tmp1 = src[i].index;
		tmp2 = src[rndi].index;
		tmp3 = src[tmp2].index;
		if (i== tmp2) continue;

		// Reassign links
		src[i].index = tmp2;
		src[rndi].index = tmp3;
		src[tmp2].index = tmp1;
	}

    sempost(startInf);
    
    // Random read
    unsigned int current = 0;
    while(*finishInference == 0){
        for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
             sum += dst[current].index;
             current = src[current].index;
             dst[current].index += src[current].index;
        }
    }

    free(src);
    free(dst);
    return sum;
}

int interference_seq(volatile uint32_t *finishInference, volatile uint32_t *startInf){
    elem_t *src;
    elem_t *dst;

    src = NULL;
    dst = NULL;

    src = (elem_t*)malloc(ARRAY_SIZE*sizeof(elem_t));
    dst = (elem_t*)malloc(ARRAY_SIZE*sizeof(elem_t));

    if(dst == NULL || src == NULL){
        perror("Error: Could not allocate buffers");
        return -1;
    }
    int sum = 0;
    printf("Init seq array\n");

	for(uint32_t i = 0; i< ARRAY_SIZE-1; i++){
		src[i].index = i+1;
        dst[i].index = i;
	}
	src[ARRAY_SIZE-1].index = 0;
	dst[ARRAY_SIZE-1].index = ARRAY_SIZE-1;
    
    sempost(startInf);
   
    // Sequential read
    unsigned int current = 0;
    while(*finishInference == 0){
        for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
             sum += dst[current].index;
             current = src[current].index;
             dst[current].index += src[current].index;
        }
    }
    free(src);
    free(dst);
    return sum;
}
