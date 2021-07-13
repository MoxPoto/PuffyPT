#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"

#include <brdfs/mixed.cuh>
#include <brdfs/lambert.cuh>
#include <brdfs/specular.cuh>
#include <brdfs/refraction.cuh>

namespace MixedBxDF {
	__device__ bool SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, vec3& attenuation, Ray& previousRay, Ray& targetRay, Object* target) {
		float diffuseProbability = 1.f - target->lighting.metalness;
		float specularProbablilty = target->lighting.metalness;
		float transmissionProbability = target->lighting.transmission;

		float sampledUniform = curand_uniform(local_rand_state);

		if (transmissionProbability >= sampledUniform) {
			RefractBRDF::SampleWorld(res, local_rand_state, pdf, extraRand, previousRay, attenuation, targetRay, target);
			return true;
		} 
		else if (specularProbablilty >= sampledUniform) {
			SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
			return true;
		}
		else if (diffuseProbability >= sampledUniform) {
			LambertBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, attenuation, targetRay, target);
			return true;
		}
		

		return false;
	}
}