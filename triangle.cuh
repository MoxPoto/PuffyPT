#ifndef TRI_H
#define TRI_H

#include "cuda_runtime.h"
#include "math.h"
#include "vec3.cuh"

namespace Tracer {
	class Triangle {
	public:
		vec3 v1;
		vec3 v2;
		vec3 v3;
		vec3 normal;

		__host__ __device__ Triangle(vec3 v1, vec3 v2, vec3 v3);
		__host__ __device__ Triangle();
	};
}

#endif