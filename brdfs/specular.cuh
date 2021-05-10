#ifndef SPECULAR_H
#define SPECULAR_H

#include "../vec3.cuh"
#include "../ray.cuh"
#include "../hitresult.cuh"
#include "../object.cuh"

#include "curand_kernel.h"

namespace Tracer {
	namespace SpecularBRDF {
		__device__ vec3 reflect(const vec3& direction, const vec3& normal);
		__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target);
	}
}

#endif