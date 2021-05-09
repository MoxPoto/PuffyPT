#include "ray.cuh"
#include "vec3.cuh"
#include "cuda_runtime.h"

namespace Tracer {
	__host__ __device__ Ray::Ray(vec3 orig, vec3 dir) {
		origin = orig;
		direction = dir;
	}
}