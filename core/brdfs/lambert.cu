#include <brdfs/lambert.cuh>
#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"
#include "math.h"
#include "math_constants.h"

#include <util/macros.h>
#include <flags/montecarlo.cuh>
#include <bluenoise.cuh>

#include <math/basic.cuh>
#include <brdfs/bxdf.cuh>

#define RANDVEC3 vec3(fmodf(curand_uniform(local_rand_state) + extraRand, 1.f),fmodf(curand_uniform(local_rand_state) + extraRand, 1.f),fmodf(curand_uniform(local_rand_state) + extraRand, 1.f))
//#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

static constexpr float M_1_PI = 0.318309886183790671538f;

// from: https://computergraphics.stackexchange.com/questions/4979/what-is-importance-sampling/4993
__device__ static float getLambertPDF(vec3 inputDir, vec3 normal) {
	return dot(inputDir, normal) * M_1_PI;
}

// from: https://computergraphics.stackexchange.com/questions/4979/what-is-importance-sampling/4993
__device__ static vec3 evaluateLambert(vec3 inputDir, vec3 normal, vec3 albedo) {
	return albedo * M_1_PI * dot(inputDir, normal);
}

namespace LambertBRDF {
	__device__ vec3 random_in_unit_sphere(curandState* local_rand_state, float extraRand) {
		vec3 p;
		
		do {
			p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
		} while (p.squared_length() >= 1.0f);
		return p;
	}
		
	__device__ void Eval(const vec3& normal, const vec3& wo, const vec3& wi, const vec3& albedo, vec3& attenuation, float& pdf) {
		attenuation = evaluateLambert(wi, normal, albedo);
		pdf = getLambertPDF(wi, normal);
	}

	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, vec3& attenuation, Ray& targetRay, Object* target, vec3 thisUV, int sampleIndex) {
		vec3 BLACK = vec3(0.f);

		targetRay.origin = res.HitPos + (res.HitNormal * 0.001f);

		float r1 = 0.0f; 
		float r2 = 0.0f;
		
		if (Flags::estimatorType == Flags::MonteCarlo::Normal) {
			r1 = curand_uniform(local_rand_state);
			r2 = curand_uniform(local_rand_state);
		}

		float r = sqrtf(r1);
		float theta = r2 * 2.f * CUDART_PI;

		float x = r * cosf(theta);
		float y = r * sinf(theta);

		// Project z up to the unit hemisphere
		float z = sqrt(1.0f - x * x - y * y);

		vec3 sampleLocalized = TransformToWorld(x, y, z, res.HitNormal);
		targetRay.direction = sampleLocalized;

		// Specular metallics and diffuse metallics come from: https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Scene/Shading.slang#L185
		float metalness = target->lighting.metalness;

		if (target->pbrMaps.mraoMap.initialized) {
			metalness = res.MRAO.b();
		}
			
		vec3 albedo = lerpVectors(res.HitAlbedo, BLACK, metalness);
		vec3 wo = vec3(0, 0, 0); // not used

		Eval(res.HitNormal, wo, sampleLocalized, albedo, attenuation, pdf);
	}

	__device__ float PDF(const HitResult& res, Object* target, const vec3& wo, const vec3& wi) {
		static const float kMinCosTheta = 1e-6f;

		if (min(wo.z(), wi.z()) < kMinCosTheta)
			return 0;

		return getLambertPDF(wi, res.HitNormal);
	}
}
