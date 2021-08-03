#ifndef LAMBERT_H
#define LAMBERT_H

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"

namespace LambertBRDF {
	__device__ vec3 random_in_unit_sphere(curandState* local_rand_state, float extraRand);
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, vec3& attenuation, Ray& targetRay, Object* target, vec3 thisUV);
	__device__ void Eval(const vec3& normal, const vec3& wo, const vec3& wi, const vec3& albedo, vec3& attenuation, float& pdf);
	__device__ float PDF(const HitResult& res, Object* target, const vec3& wo, const vec3& wi);

}


#endif