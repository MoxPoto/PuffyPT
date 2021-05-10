#ifndef OBJECT_H
#define OBJECT_H

#include "cuda_runtime.h"
#include "triangle.cuh"
#include "vec3.cuh"
#include "hitresult.cuh"
#include "ray.cuh"

namespace Tracer {
	enum BRDF {
		Lambertian
	};

	class Object {
	public:
		vec3 color = vec3(1, 1, 1);
		float emission = 1.f;
		BRDF matType = BRDF::Lambertian;

		__host__ __device__ Object();

		__host__ __device__ bool virtual tryHit(const Ray& ray, float closest, HitResult& result);
	};
}
#endif