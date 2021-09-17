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
#include <classes/lighting.cuh>
#include <brdfs/bxdf.cuh>

class Object {
public:
	vec3 color = vec3(1, 1, 1);
	vec3 position = vec3(0, 0, 0);
	int objectID = 0;
	float emission = 1.f;
	BxDF* shading = nullptr;

	// TODO:
	// deprecate the usage of specific BRDFs, this isn't even considered in the pathtracer anymore
	// all lighting choices are selected via the Mixed BxDF
	BRDF matType = BRDF::Lambertian;
	LightingOptions lighting;
	glm::mat3x3 transform;
	Texture texture;
	PBRMap pbrMaps;

	__host__ __device__ Object();

	__device__ vec3 GetColor(const HitResult& rayThatHit);
	__device__ bool virtual TryHit(const Ray& ray, HitResult& result);
	__device__ bool virtual AnyHit(const Ray& ray);
};

#endif

