#ifndef HDRI_H
#define HDRI_H
#include "../vec3.cuh"

namespace Tracer {
	class HDRI {
	public:
		int resX = 0;
		int resY = 0;
		float brightness = 1.f;

		__device__ HDRI();
		__device__ float getPitch(const vec3& N);
		__device__ float getYaw(const vec3& N);
		__device__ vec3 getPixel(const int& x, const int& y, float* imagePtr);
		__device__ vec3 getPixelFromRay(const vec3& vec, float* imagePtr);
	};
}

#endif