// blue noise related definitions

#include <images/texture.cuh>
#include <classes/vec3.cuh>

#include "math_constants.h"

namespace Bluenoise {
	extern __device__ Texture blueNoiseTex;
	extern __device__ const vec3 blueNoiseDisk[64];
	extern __device__ const float goldenRatio;

	extern __device__ int frameNumber;

	// so simple!!
	extern __device__ vec3 CalculateSample(int sampleIndex, vec3 uv);
}