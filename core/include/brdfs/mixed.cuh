#ifndef MIXED_H
#define MIXED_H

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"

namespace MixedBxDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, vec3& attenuation, Ray& previousRay, Ray& targetRay, Object* target);
}

#endif