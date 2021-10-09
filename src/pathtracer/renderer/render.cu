#include <renderer/render.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void renderKernel(float* framebuffer, DWORD* dxFramebuffer, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (x >= width || y >= height)
		return;

	DWORD newCol = D3DCOLOR_XRGB(0, 255, 0);
	int pixel_index = y * width + x;

	dxFramebuffer[pixel_index] = newCol;
}