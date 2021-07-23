#ifndef SPECULAR_H
#define SPECULAR_H

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"

namespace SpecularBRDF {
	__device__ vec3 reflect(const vec3& direction, const vec3& normal);
	__device__ float schlick(float cosine, float ref_idx);
	__device__ bool SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target);
	__device__ void Eval(float alpha, float metalness, Object* target, const vec3& normal, const vec3& wo, const vec3& wi, const vec3& albedo, vec3& attenuation, float& pdf);
	__device__ float PDF(const HitResult& res, Object* target, const vec3& wo, const vec3& wi);
}


#endif