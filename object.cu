#include "object.cuh"
#include "ray.cuh"
#include "hitresult.cuh"
#include "vec3.cuh"

namespace Tracer {
	__host__ __device__ Object::Object() {
		color = vec3(0, 0, 0);
		emission = 1.f;
	}

	__host__ __device__ vec3 Object::getColor(const HitResult& rayThatHit) {
		if (!texture.initialized) 
			return texture.GetPixel(rayThatHit.u, rayThatHit.v);

		return texture.GetPixel(rayThatHit.u, rayThatHit.v) * color;
	}

	__host__ __device__ bool Object::tryHit(const Ray& ray, HitResult& result) {
		return false;
	}

	__host__ __device__ bool Object::anyHit(const Ray& ray) {
		return true; // if something simply just returns "true" on the anyhit pass it's pretty much safe to assume it's not accelerated
	}
}