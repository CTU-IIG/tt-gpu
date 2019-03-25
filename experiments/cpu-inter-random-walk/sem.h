#ifndef SEM_H
#define SEM_H
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((always_inline)) inline void semwait(volatile uint32_t *count) {
	uint32_t old_value;
	while(1){
		old_value = *count;
		if(old_value == 0){
			continue;
		}
		if(__sync_bool_compare_and_swap(count, old_value, old_value - 1)){
			break;
		}
	}
	__asm volatile("dmb sy":::"memory");
}

__attribute__((always_inline)) inline void sempost(volatile uint32_t *count) {
	uint32_t old_value;
	do{
		old_value = *count;
		__asm volatile("dmb sy":::"memory");
	}while (!__sync_bool_compare_and_swap(count, old_value, old_value + 1));
	__asm volatile("dmb sy":::"memory");
}
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
