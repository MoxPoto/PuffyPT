#ifndef RAY_H
#define RAY_H

#include "cuda_runtime.h"
#include "vec3.cuh"

namespace Tracer {
	class Ray {
	public:
		vec3 origin;
		vec3 direction;

		__host__ __device__ Ray(vec3 origin, vec3 direction);
	};
}

#endif