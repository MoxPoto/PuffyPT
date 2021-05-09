#include "hitresult.cuh"
#include "vec3.cuh"

namespace Tracer {
	__host__ __device__ HitResult::HitResult() {
		HitPos = vec3();
		HitNormal = vec3();

		t, u, v = 0, 0, 0;
	}
}