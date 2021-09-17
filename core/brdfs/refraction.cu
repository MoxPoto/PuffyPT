#include <classes/vec3.cuh>
#include <classes/hitresult.cuh>
#include "curand_kernel.h"
#include <classes/ray.cuh>
#include <classes/object.cuh>

#include <brdfs/specular.cuh>
#include <brdfs/refraction.cuh>

#include <math/basic.cuh>
#include <math/ggx.cuh>
#include "math_constants.h"

/*
local function refract(I, N, ior)
	local cosI = math.clamp(-1, 1, I:dot(N))
	local etaI, etaT = 1, ior

	if cosI < 0 then
		cosI = -cosI
	else
		etaT, etaI = etaI, ior
		N = -N
	end

	local eta = etaI / etaT
	local k = 1 - eta^2 * (1 - cosI^2)
	return k < 0 and 0 or eta * I + (eta * cosI - math.sqrt(k)) * N
end
*/

__device__ static vec3 calculateBeersLaw(vec3 color, float distanceInsideObject) {
	// multiplier = e^(-color * distanceInsideObject)
	vec3 result = -color * distanceInsideObject;
	return vec3(expf(result.x()), expf(result.y()), expf(result.z()));
}

__device__ static void swap(float& a, float& b) {
	float temp = a;
	a = b;
	b = temp;
}

__device__ static vec3 refract(vec3 incidence, vec3 normal, float ior, bool& invalid) {
	float cosi = clamp(-1, 1, dot(incidence, normal));
	float etai = 1, etat = ior;
	vec3 n = normal;

	if (cosi < 0.f) { 
		cosi = -cosi; 
	} else { 
		swap(etai, etat);
		n = -normal; 
	}

	float eta = etai / etat;
	float k = 1.f - eta * eta * (1.f - cosi * cosi);

	invalid = (k < 0.f);

	return k < 0.f ? vec3(0, 0, 0) : eta * incidence + (eta * cosi - sqrtf(k)) * n;
}


namespace RefractBRDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float& pdf, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target, BRDF& brdfChosen) {
		vec3 wo = -previousRay.direction;
		vec3 wi = reflect(previousRay.direction, res.HitNormal);

		float uniform = curand_uniform(local_rand_state);
		float currentIOR = target->lighting.ior; //res.backface ? 1.00f : target->lighting.ior;
		vec3 normal = res.backface ? -res.HitNormal : res.HitNormal;

		float fresnel = schlick(dot(-previousRay.direction, res.HitNormal), currentIOR); 
		bool outside = !res.backface;

		if (uniform <= fresnel) {
			// Take reflection path, this is usually when we're experiencing total internal reflection or just, normal fresnel
			SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
			brdfChosen = BRDF::Specular;
		}
		else {
			bool invalidRefract = false;
			/*
			// Do some refraction

			float u1 = curand_uniform(local_rand_state);
			float u2 = curand_uniform(local_rand_state);
			// random 1 and 2 in the cook-torrance paper
			float metalness = target->lighting.metalness;
			float roughness = target->lighting.roughness;

			if (target->pbrMaps.mraoMap.initialized) {
				metalness = res.MRAO.b();
				roughness = res.MRAO.g();

				// Might be a good idea in the future to choose a specific
				// MRAO format before just assuming that metalness = b and roughness = g
			}

			bool outside = dot(-wo, normal) < 0.f;

			float alpha = fmaxf(0.001f, roughness * roughness);
			static const float kMinCosTheta = 1e-4f;

			float thetaM = atanf((alpha * sqrtf(u1)) / sqrt(1.f - u1));
			float phiM = (2.f * static_cast<float>(CUDART_PI) * u2);

			vec3 m = TransformToWorld(sinf(thetaM) * cosf(phiM), sinf(thetaM) * sinf(phiM), cosf(thetaM), normal);
			m.make_unit_vector();
		
			float cosi = clamp(-1, 1, dot(-wo, normal));
			float etai = 1, etat = currentIOR;

			if (cosi < 0.f) {
				cosi = -cosi;
			}
			else {
				swap(etai, etat);
				// n = -normal;
			}

			float c = dot(wo, m);
			float nIor = etai / etat;

			float k = 1.f - nIor * nIor * (1.f - cosi * cosi);

			invalidRefract = (k < 0.f);

			vec3 newDir = (nIor * c - sign(dot(wo, normal)) * sqrtf(1.0f + nIor * (c * c - 1.0f))) * m - (nIor * wo);
			*/
			vec3 refractionDir = unit_vector(refract(-wo, normal, currentIOR, invalidRefract));
			vec3 refractionOrigin = res.backface ? res.HitPos + (normal * 0.01f) : res.HitPos - (normal * 0.01f);

			if (invalidRefract) {
				// 2nd case Total Internal Reflection
				//SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
				
				targetRay.direction = reflect(-wo, normal);
				targetRay.origin = refractionOrigin;

				attenuation = vec3(1, 1, 1);
				pdf = 1;

				brdfChosen = BRDF::Specular;

				return;
			}
			/*
			float term1 = (abs(dot(wo, m)) * abs(dot(wi, m))) / (abs(dot(wo, normal)) * abs(dot(wi, normal)));
			
			float numerator = (etat * etat) * (1.f - schlick(dot(wo, m), currentIOR)) * GGXDistribution(roughness, thetaM, normal, m);
			float denominator = powf((etai * dot(wo, m)) + (etat * (dot(wi, m))), 2);
			*/

			targetRay.direction = refractionDir;
			targetRay.origin = refractionOrigin;

			// Calculate the color for this ray
			attenuation = vec3(1, 1, 1);
			pdf = 1;

			brdfChosen = BRDF::Refraction;
		}
	}
}