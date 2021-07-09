﻿#ifndef SPECULAR_H
#define SPECULAR_H

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"

namespace SpecularBRDF {
	__device__ vec3 reflect(const vec3& direction, const vec3& normal);
	__device__ float schlick(float cosine, float ref_idx);
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target);
}


#endif