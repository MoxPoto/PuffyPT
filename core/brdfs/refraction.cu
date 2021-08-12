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


__device__ static bool refract(vec3 incidence, vec3 normal, float ior, vec3& outputVector) {
	float cosI = clamp(-1.f, 1.f, dot(unit_vector(incidence), normal));
	float etaI = 1.f;
	float etaT = ior;

	if (cosI < 0.f) {
		cosI = -cosI;
	}
	else {
		etaT = etaI;
		etaI = ior;
		normal = -normal;
	}

	float eta = etaI / etaT;
	float k = 1.f - powf(eta, 2) * (1.f - powf(cosI, 2));

	if (k <= 0.f) {
		return false;
	} else {
		outputVector = eta * unit_vector(incidence) + (eta * cosI - sqrtf(k)) * normal;
		return true;
	}
}


namespace RefractBRDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float& pdf, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target, BRDF& brdfChosen) {
		using SpecularBRDF::schlick;
		using SpecularBRDF::reflect;


		float uniform = curand_uniform(local_rand_state);
		float currentIOR = target->lighting.ior;

		float fresnel = schlick(dot(-previousRay.direction, res.HitNormal), currentIOR) * 2.5f;

		vec3 normal = res.backface ? -res.HitNormal : res.HitNormal;

		// Generation for a cook torrance BTDF direction relies on a microfacet, so we're going to generate one with spherical coordinates using the GGX format

		vec3 wo = -previousRay.direction;
		vec3 wi = reflect(previousRay.direction, res.HitNormal);


		if (uniform <= fresnel) {
			// Take reflection path

			SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
			brdfChosen = BRDF::Specular;
		}
		else {

			float u1 = curand_uniform(local_rand_state);
			float u2 = curand_uniform(local_rand_state);
			// random 1 and 2 in the cook-torrance paper
			float roughness = target->lighting.roughness;

			if (target->pbrMaps.mraoMap.initialized) {
				roughness = res.MRAO.g();

				// Might be a good idea in the future to choose a specific
				// MRAO format before just assuming that metalness = b and roughness = g
			}

			float alpha = fmaxf(0.001f, roughness * roughness);
			static const float kMinCosTheta = 1e-6f;

			float thetaM = atanf((alpha * sqrtf(u1)) / sqrt(1.f - u1));
			float phiM = (2.f * static_cast<float>(CUDART_PI) * u2);

			vec3 m = TransformToWorld(sinf(thetaM) * cosf(phiM), sinf(thetaM) * sinf(phiM), cosf(thetaM), res.HitNormal);
			m.make_unit_vector();

			if (dot(wo, m) < kMinCosTheta) {
				// TODO:
				// Make sure refraction can give out invalid samples (since it can)
				//return false;
			}

			// Generating the new out direction is noted in equation 40 at
			// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

			float c = dot(wo, m);
			float nI = 1.0;
			float nT = currentIOR;
			float cosI = clamp(-1.f, 1.f, dot(wo, normal));
			
			if (cosI < 0.0f) {
				// Switch IORs before computing the final N
				nI, nT = nT, nI;
			}
			

			float n = nI / nT;

			vec3 newDir = (n * c - sign(dot(wo, normal)) * sqrtf(1.0f + n * (c * c - 1.0f))) * m - n * wo;

			targetRay.origin = res.backface ? res.HitPos + normal * 0.01f : res.HitPos - normal * 0.01f;

			vec3 ht = -(nI * wo + nT * newDir);
			// ht is defined as a function that normalizes itself
			ht.make_unit_vector();

			// Refraction term
			float term1 = (fabsf(dot(wo, ht)) * fabsf(dot(newDir, ht))) / (fabsf(dot(wo, normal)) * fabsf(dot(newDir, normal)));

			// term 2
			float noSquared = nT * nT;
			/*
			float numerator = noSquared * (1.0f - schlick(dot(wo, ht), currentIOR) * GGXGeometry(wo, normal, m, alpha) * GGXDistribution(alpha, thetaM, normal, m));
			float denominator = nI * (dot(wo, ht)) + nT * (dot(newDir, ht));
			denominator = denominator * denominator; // Square the denominator

			float finalTerm = term1 * (numerator / denominator);

			targetRay.direction = unit_vector(newDir);

			pdf = 1.f;// GGXDistribution(alpha, thetaM, res.HitNormal, m)* fabsf(dot(m, res.HitNormal));
			attenuation = vec3(0, 0, 1);
			*/
		}

		/*

		if (uniform <= fresnel) {
			// Take reflection path

			targetRay.origin = res.HitPos + (res.HitNormal * 0.001f);
			targetRay.direction = reflect(previousRay.direction, res.HitNormal);

			brdfChosen = BRDF::Specular;

			float weight = (fresnel);
			pdf = 1;

			attenuation = vec3(fresnel, fresnel, fresnel)


			SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
			brdfChosen = BRDF::Specular;
		}
		else {
			// Take refraction path


			targetRay.origin = res.backface ? res.HitPos + normal * 0.01f : res.HitPos - normal * 0.01f;

			bool refracted = refract(previousRay.direction, normal, currentIOR, targetRay.direction);

			if (!refracted) {
				SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
				brdfChosen = BRDF::Specular;
			}
			else {
				// Calculate roughness
				vec3 roughnessDir = normal + vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
				roughnessDir.make_unit_vector();

				float roughness = target->lighting.roughness;
				if (target->pbrMaps.mraoMap.initialized) {
					roughness = res.MRAO.g();
				}

				if (roughness > 0.0f) {
					targetRay.direction = lerpVectors(targetRay.direction, roughnessDir, roughness * roughness);
				}

				brdfChosen = BRDF::Refraction;
			}

			if (res.backface) {
				attenuation = calculateBeersLaw(1.f - res.HitAlbedo, res.t);
			}

			float weight = (1.f - fresnel);
			pdf = weight;


		}

	}
	*/
	}
}