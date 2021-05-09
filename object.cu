#include "object.cuh"
#include "ray.cuh"
#include "hitresult.cuh"
#include "vec3.cuh"

namespace Tracer {
	__host__ __device__ Object::Object() {
		color = vec3(0, 0, 0);
		emission = 1.f;
	}

	__host__ __device__ bool Object::tryHit(const Ray& ray, float closest, HitResult& result) {
		return false;
	}
}