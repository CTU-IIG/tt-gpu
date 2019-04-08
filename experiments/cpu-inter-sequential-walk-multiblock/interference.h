#ifndef INERFERENCE_H
#define INTERFERENCE_H


int interference_random(volatile uint32_t *finishInference, volatile uint32_t *startInf);

int interference_seq(volatile uint32_t *finishInference, volatile uint32_t *startInf);
#endif
