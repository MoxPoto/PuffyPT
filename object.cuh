﻿#ifndef OBJECT_H
#define OBJECT_H

#include "cuda_runtime.h"
#include "triangle.cuh"
#include "vec3.cuh"
#include "hitresult.cuh"
#include "ray.cuh"
#include "images/texture.cuh"

#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Tracer {
	enum BRDF {
		Lambertian,
		Specular,
		Refraction
	};

	struct LightingOptions {
		float roughness = 0.0f;
		float ior = 2.41f;
	};

	class Object {
	public:
		vec3 color = vec3(1, 1, 1);
		vec3 position = vec3(0, 0, 0);
		int objectID = 0;
		float emission = 1.f;
		BRDF matType = BRDF::Lambertian;
		LightingOptions lighting;
		glm::mat3x3 transform;
		Texture texture;

		__host__ __device__ Object();

		__device__ vec3 getColor(const HitResult& rayThatHit);
		__host__ __device__ bool virtual tryHit(const Ray& ray, HitResult& result);
		__host__ __device__ bool virtual anyHit(const Ray& ray);
	};
}
#endif

