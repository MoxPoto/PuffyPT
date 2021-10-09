#ifndef PATHTRACER_CUH
#define PATHTRACER_CUH

#include <Windows.h>
#include <cuda_runtime.h>
#include <vector>

#include <mutex>

// To be used within the Pathtracer class
#define checkCudaErrors(val) ErrorCheck( (val), #val, __FILE__, __LINE__ )

class Pathtracer {
private:
	float* framebuffer; // Normal, 0-inf framebuffer

	std::mutex updateMutex; // Mutex to prevent buffers from being remove mid-render

	bool valid = true;
	
	int width;
	int height;

	std::vector<void*> buffersToRelease; // Buffers to free when destructor is called
public:
	DWORD* dxFramebuffer; // DirectX, DWORD framebuffer

	__host__ void Update();

	__host__ void Allocate(void* gpuMemory, size_t bufferSize, bool managed);
	__host__ void ImGuiUpdate();
	__host__ void ErrorCheck(cudaError_t err, char const* const func, const char* const file, int const line);

	__host__ Pathtracer(int _width, int _height);
	__host__ ~Pathtracer();
};

#endif