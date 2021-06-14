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

		float u1, u2, u3;
		float vt1, vt2, vt3; 

		__device__ Triangle(vec3 v1q, vec3 v2q, vec3 v3q, float _u1, float _v1, float _u2, float _v2, float _u3, float _v3);
		__device__ Triangle();
	};
}

#endif