#include "../vec3.cuh"
#include "../hitresult.cuh"
#include "curand_kernel.h"
#include "../ray.cuh"
#include "../object.cuh"

#include "specular.cuh"
#include "refraction.cuh"

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

using namespace Tracer;

__device__ static float clamp(float d, float min, float max) {
	const float t = d < min ? min : d;
	return t > max ? max : t;
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

	if (k < 0.f) {
		return false;
	} else {
		outputVector = eta * incidence + (eta * cosI - sqrtf(k)) * normal;
		return true;
	}
}

namespace Tracer {
	namespace RefractBRDF {
		__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target) {
			using SpecularBRDF::schlick;

			float uniform = curand_uniform(local_rand_state);

			float fresnel = schlick(dot(-previousRay.direction, res.HitNormal), target->lighting.ior);
			/*
			if (5 == 5) {
				attenuation = vec3(fresnel, fresnel, fresnel);
				return;
			}
			*/

			if (uniform <= fresnel) {
				// Take reflection path
				SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, previousRay, attenuation, targetRay, target);

				attenuation /= fresnel;
			}
			else {
				// Take refraction path

				
				targetRay.origin = res.backface ? res.HitPos + (-res.HitNormal * 0.01f) : res.HitPos - (res.HitNormal * 0.01f);


				bool refracted = refract(previousRay.direction, res.HitNormal, target->lighting.ior, targetRay.direction);

				if (!refracted) {
					SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, previousRay, attenuation, targetRay, target);
				}
				
				attenuation /= (1.f - fresnel);
			}

		}
	}
}