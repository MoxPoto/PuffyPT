#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.cuh"
#include "object.cuh"
#include "hitresult.cuh"
#include "ray.cuh"

namespace Tracer {
	class Sphere : public Object {
	public:
		float radius = .2f;
		vec3 center = vec3(0, 0, 0);

		__host__ __device__ Sphere(vec3 position, float radius);
		__host__ __device__ bool virtual tryHit(const Ray& ray, HitResult& result);
	};
}

#endif