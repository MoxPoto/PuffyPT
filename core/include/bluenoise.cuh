// blue noise related definitions

#include <images/texture.cuh>
#include <classes/vec3.cuh>

#include "math_constants.h"

namespace Bluenoise {
	extern __device__ Texture* blueNoiseTex;
	extern __device__ vec3 blueNoiseDisk[64];
	extern __device__ const float goldenRatio;
	extern __device__ bool initialized;

	extern __device__ int frameNumber;
	extern __device__ int MAX_FRAMES;

	// so simple!!
	extern __device__ vec3 CalculateSample(int sampleIndex, vec3 uv);
	extern __device__ void Initialize();
}

extern __host__ bool LoadBluenoise(const char* path);