#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"

#include <brdfs/lambert.cuh>
#include <brdfs/specular.cuh>

// TODO:
// Integrate refraction into the BxDF
namespace MixedBxDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, vec3& attenuation, Ray& previousRay, Ray& targetRay, Object* target) {
		float diffuseProbability = 1.f - target->lighting.metalness;

		if (diffuseProbability >= curand_uniform(local_rand_state)) {
			LambertBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, attenuation, targetRay, target);
		}
		else {
			SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
		}
	}
}