#include <classes/vec3.cuh>
#include <classes/hitresult.cuh>
#include "curand_kernel.h"
#include <classes/ray.cuh>
#include <classes/object.cuh>

#include <brdfs/specular.cuh>
#include <brdfs/refraction.cuh>

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

__device__ static inline float lerp(float a, float b, float f)
{
	return (a * (1.0f - f)) + (b * f);
}

__device__ static inline vec3 lerpVectors(vec3 a, vec3 b, float f) {
	return vec3(
		lerp(a.x(), b.x(), f),
		lerp(a.y(), b.y(), f),
		lerp(a.z(), b.z(), f)
	);
}

__device__ static float clamp(float d, float min, float max) {
	const float t = d < min ? min : d;
	return t > max ? max : t;
}

__device__ static vec3 calculateBeersLaw(vec3 color, float distanceInsideObject) {
	// multiplier = e^(-color * distanceInsideObject)
	vec3 result = -color * distanceInsideObject;
	return vec3(expf(result.x()), expf(result.y()), expf(result.z()));
}

__device__ static bool refract(vec3 incidence, vec3 normal, float ior, vec3& outputVector) {
	float cosI = clamp(-1.f, 1.f, dot(incidence, normal));
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
		outputVector = eta * incidence + (eta * cosI - sqrtf(k)) * normal;
		return true;
	}
}

namespace RefractBRDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float& pdf, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target, BRDF& brdfChosen) {
		using SpecularBRDF::schlick;
		using SpecularBRDF::reflect;

		float uniform = curand_uniform(local_rand_state);
		float currentIOR = res.backface ? 1.00f : target->lighting.ior;

		float fresnel = schlick(dot(-previousRay.direction, res.HitNormal), currentIOR);

		

		/*
		if (5 == 5) {
			attenuation = vec3(fresnel, fresnel, fresnel);
			return;
		}
		*/

		if (uniform <= fresnel) {
			// Take reflection path
			targetRay.origin = res.HitPos + (res.HitNormal * 0.001f);
			targetRay.direction = reflect(previousRay.direction, res.HitNormal);

			brdfChosen = BRDF::Specular;

			float weight = (fresnel);
			pdf = weight;
		}
		else {
			// Take refraction path

				
			targetRay.origin = res.backface ? res.HitPos + (-res.HitNormal * 0.01f) : res.HitPos - (res.HitNormal * 0.01f);

				
			bool refracted = refract(previousRay.direction, res.HitNormal, currentIOR, targetRay.direction);

			if (!refracted) {
				targetRay.origin = res.HitPos + (res.HitNormal * 0.001f);
				targetRay.direction = reflect(previousRay.direction, res.HitNormal);

				brdfChosen = BRDF::Specular;
			}
			else {
				// Calculate roughness
				vec3 roughnessDir = res.HitNormal + vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
				roughnessDir.make_unit_vector();

				float roughness = target->lighting.roughness;
				if (target->pbrMaps.mraoMap.initialized) {
					roughness = res.MRAO.g();
				}

				targetRay.direction = lerpVectors(targetRay.direction, roughnessDir, roughness * roughness);

				brdfChosen = BRDF::Refraction;
			}

			if (res.backface) {
				attenuation = calculateBeersLaw(1.f - res.HitAlbedo, res.t);
			}
				
			float weight = (1.f - fresnel);
			pdf = weight;

			
		}

	}
}
