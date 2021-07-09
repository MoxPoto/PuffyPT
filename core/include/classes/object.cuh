#ifndef OBJECT_H
#define OBJECT_H

#include "cuda_runtime.h"
#include <classes/triangle.cuh>
#include <classes/vec3.cuh>
#include <classes/hitresult.cuh>
#include <classes/ray.cuh>
#include <images/texture.cuh>

#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>


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

	__device__ vec3 GetColor(const HitResult& rayThatHit);
	__host__ __device__ bool virtual TryHit(const Ray& ray, HitResult& result);
	__host__ __device__ bool virtual AnyHit(const Ray& ray);
};

#endif

