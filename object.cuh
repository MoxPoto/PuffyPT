#ifndef OBJECT_H
#define OBJECT_H

#include "cuda_runtime.h"
#include "triangle.cuh"
#include "vec3.cuh"
#include "hitresult.cuh"
#include "ray.cuh"

namespace Tracer {
	enum BRDF {
		Lambertian,
		Specular
	};

	struct LightingOptions {
		float roughness = 0.0f;
		float ior = 1.1f;
	};

	class Object {
	public:
		vec3 color = vec3(1, 1, 1);
		vec3 position = vec3(0, 0, 0);
		int objectID = 0;
		float emission = 1.f;
		BRDF matType = BRDF::Lambertian;
		LightingOptions lighting;

		__host__ __device__ Object();

		__host__ __device__ bool virtual tryHit(const Ray& ray, HitResult& result);
		__host__ __device__ bool virtual anyHit(const Ray& ray);
	};
}
#endif

