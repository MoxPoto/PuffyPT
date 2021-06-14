#include "triangle.cuh"
#include "cuda_runtime.h"
#include "math.h"
#include "vec3.cuh"

namespace Tracer {
	__device__ Triangle::Triangle(vec3 v1q, vec3 v2q, vec3 v3q, float _u1, float _v1, float _u2, float _v2, float _u3, float _v3) {
		v1 = v1q;
		v2 = v2q;
		v3 = v3q;
		
		u1 = _u1;
		u2 = _u2;
		u3 = _u3;

		vt1 = _v1;
		vt2 = _v2;
		vt3 = _v3;

		vec3 theU = (v2 - v1);
		vec3 theV = (v3 - v1);
		
		normal = cross(theV, theU);
		normal.make_unit_vector(); // wtf

	}

	__device__ Triangle::Triangle() {
		v1 = vec3();
		v2 = vec3();
		v3 = vec3();

		u1 = 0.0f;
		u2 = 0.0f;
		u3 = 0.0f;

		vt1 = 0.0f;
		vt2 = 0.0f;
		vt3 = 0.0f;

		normal = vec3();
	}
}