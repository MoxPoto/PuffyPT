#ifndef LAMBERT_H
#define LAMBERT_H

#include "../vec3.cuh"
#include "../ray.cuh"
#include "../hitresult.cuh"
#include "../object.cuh"

#include "curand_kernel.h"

namespace Tracer {
	namespace LambertBRDF {
		__device__ vec3 random_in_unit_sphere(curandState* local_rand_state);
		__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, vec3& attenuation, Ray& targetRay, Object* target);
	}
}

#endif