#ifndef HDRI_H
#define HDRI_H
#include "../vec3.cuh"

namespace Tracer {
	class HDRI {
	public:
		unsigned char* imageData;
		int resX = 0;
		int resY = 0;
		
		__host__ __device__ HDRI();
		__host__ __device__ void loadData(unsigned char* dataToLoad);
		__host__ __device__ float getPitch(const vec3& N);
		__host__ __device__ float getYaw(const vec3& N);
		__host__ __device__ vec3 getPixel(const int& x, const int& y);
		__host__ __device__ vec3 getPixelFromRay(const vec3& vec);
	};
}

#endif