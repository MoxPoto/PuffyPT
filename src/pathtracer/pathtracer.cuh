#ifndef PATHTRACER_CUH
#define PATHTRACER_CUH

#include <Windows.h>
#include <cuda_runtime.h>
#include <vector>

class Pathtracer {
private:
	float* framebuffer; // Normal, 0-1 framebuffer
	DWORD* dxFramebuffer; // DirectX, DWORD framebuffer

	int width;
	int height;

	std::vector<void*> buffersToRelease; // Buffers to free when destructor is called
public:
	__host__ void Allocate(void* gpuMemory);
	__host__ void ImGuiUpdate();

	__host__ Pathtracer();
};

#endif