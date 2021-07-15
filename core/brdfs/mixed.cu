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
		using SpecularBRDF::reflect;

		float specularProbablilty = 1.f - target->lighting.roughness;
		float diffuseProbability = target->lighting.roughness;

		if (target->pbrMaps.mraoMap.initialized) {
			specularProbablilty = 1.f - res.MRAO.g();
			diffuseProbability = res.MRAO.g();
		}
		float transmissionProbability = 0.f; // temporary

		float sampledUniform = curand_uniform(local_rand_state);

		vec3 wo = -previousRay.direction;
		vec3 wi = reflect(previousRay.direction, res.HitNormal);
		
		// Most of this comes from: https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Experimental/Scene/Material/BxDF.slang#L682-L708
		if (sampledUniform < diffuseProbability) {
			LambertBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, attenuation, targetRay, target);

			pdf *= diffuseProbability;

			if (specularProbablilty > 0)
				pdf += specularProbablilty * SpecularBRDF::PDF(res, target, wo, wi);

			return true;
		}
		else if (sampledUniform < diffuseProbability + specularProbablilty) {
			bool valid = SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
			pdf *= specularProbablilty;

			if (diffuseProbability > 0)
				pdf += diffuseProbability * LambertBRDF::PDF(res, target, wo, wi);

			return valid;
		}

		return false;
	}
}