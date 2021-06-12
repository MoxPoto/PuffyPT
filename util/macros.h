#include <stdio.h>


#define NULLPTR_HIT(msg) printf("==================================\n"); printf("            NULLPTR!\n"); printf("%s says:\n%s\n", __FUNCTION__, msg); printf("==================================\n");
#define GPU_DEBUG(msg, ...) printf("[gpu %s]: ", __FUNCTION__); printf(msg, __VA_ARGS__); printf("\n");
#define HOST_DEBUG(msg, ...) printf("[host %s]: ", __FUNCTION__); printf(msg, __VA_ARGS__); printf("\n");
#define min(a,b) ((a)<(b)?(a):(b))
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )
#define max(a,b) ((a)>(b)?(a):(b))