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

__device__ static vec3 refract(vec3 incidence, vec3 normal, float ior) {
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

	return k < 0.f ? vec3(0, 0, 0) : eta * incidence + (eta * cosi - sqrtf(k)) * n;
}


namespace RefractBRDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float& pdf, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target, BRDF& brdfChosen) {
		using SpecularBRDF::schlick;
		using SpecularBRDF::reflect;

		vec3 wo = -previousRay.direction;

		float uniform = curand_uniform(local_rand_state);
		float currentIOR = target->lighting.ior; //res.backface ? 1.00f : target->lighting.ior;
		vec3 normal = res.backface ? -res.HitNormal : res.HitNormal;

		float fresnel = schlick(dot(wo, normal), currentIOR);
		bool outside = dot(-wo, normal) < 0.f;

		if (uniform <= fresnel) {
			// Take reflection path, this is usually when we're experiencing total internal reflection or just, normal fresnel
			SpecularBRDF::SampleWorld(res, local_rand_state, extraRand, pdf, previousRay, attenuation, targetRay, target);
			brdfChosen = BRDF::Specular;
		}
		else {
			// Do some refraction
			vec3 refractionDir = unit_vector(refract(-wo, normal, currentIOR));
			vec3 refractionOrigin = outside ? res.HitPos - (normal * 0.01f) : res.HitPos + (normal * 0.01f);

			targetRay.direction = refractionDir;
			targetRay.origin = refractionOrigin;

			// Calculate the color for this ray
			attenuation = vec3(1, 1, 1); // calculateBeersLaw(1.f - res.HitAlbedo, res.t);
			// pdf = (1.f - fresnel);
		}
	}
}