#include <stdio.h>


#define NULLPTR_HIT(msg) printf("==================================\n"); printf("            NULLPTR!\n"); printf("%s\n", msg); printf("==================================\n");
#define GPU_DEBUG(msg, ...) printf("[gpu]: "); printf(msg, __VA_ARGS__);
#define HOST_DEBUG(msg, ...) printf("[host]: "); printf(msg, __VA_ARGS__);
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))