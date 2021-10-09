#include <cuda_runtime.h>
#include <d3d9.h>

extern __global__ void renderKernel(float* framebuffer, DWORD* dxFramebuffer, int width, int height);