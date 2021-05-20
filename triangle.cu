#include "triangle.cuh"
#include "cuda_runtime.h"
#include "math.h"
#include "vec3.cuh"

namespace Tracer {
	__host__ __device__ Triangle::Triangle(vec3 v1q, vec3 v2q, vec3 v3q) {
		v1 = v1q;
		v2 = v2q;
		v3 = v3q;
		
		vec3 theU = (v2 - v1);
		vec3 theV = (v3 - v1);
		
		normal = cross(theV, theU);
		normal.make_unit_vector(); // wtf

	}

	__host__ __device__ Triangle::Triangle() {
		v1 = vec3();
		v2 = vec3();
		v3 = vec3();

		normal = vec3();
	}
}