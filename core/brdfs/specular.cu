#include <brdfs/specular.cuh>
#include <brdfs/lambert.cuh>

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"
#include "math_constants.h"

#include <util/macros.h>

#include <math/ggx.cuh>
#include <math/basic.cuh>
#define PROTECTZERO(statement) fmaxf(0.001f, statement)

__device__ static vec3 coloredSchlick(vec3 r0, float cosine, float ref_idx) {
	return r0 + (vec3(1.0f) - r0) * powf(fmaxf((1.0f - cosine), 0), 5.0f);
}

__device__ static float FalcorNDFGGX(float alpha, float cosTheta) {
	float a2 = alpha * alpha;
	float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
	return a2 / (d * d * static_cast<float>(CUDART_PI));
}

namespace SpecularBRDF {
	__device__ vec3 reflect(const vec3& direction, const vec3& normal) {
		return direction - 2.0f * dot(direction, normal) * normal;
	}
	
	__device__ float schlick(float cosine, float ref_idx) {
		float r0 = (ref_idx - 1.0f) / (ref_idx + 1.0f);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
	}

	__device__ void Eval(float alpha, float metalness, Object* target, const vec3& normal, const vec3& wo, const vec3& wi, const vec3& albedo, vec3& attenuation, float& pdf) {
		vec3 m = unit_vector((wi + wo));

		float thetaM = thetaFromVec(m);

		pdf = GGXDistribution(alpha, thetaM, normal, m) * fabsf(dot(m, normal));

		// evaluate cook-torrance

		float f = fabsf((1.f - target->lighting.ior) / (1.f + target->lighting.ior));
		// in my schlick's function, f0 is represented as r0
		float F0 = (f * f);

		vec3 finalSchlicksInput = lerpVectors(vec3(F0), albedo, metalness);

		vec3 fresnelTerm = coloredSchlick(finalSchlicksInput, dot(wo, m), target->lighting.ior);
		/*
		vec3 numerator = fresnelTerm * GGXDistribution(alpha, thetaM, normal, m) * GGXGeometry(wi, normal, m, alpha);
		attenuation = numerator;
		*/

	}

	__device__ bool SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target) {
		// wo = -previousRay.direction;
		// wi = reflect(-wo, hitnormal);

		// some of the code is based on: https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/shaders.cpp#L63

		vec3 wo = -previousRay.direction;
		vec3 wi = reflect(previousRay.direction, res.HitNormal);
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

		float alpha = fmaxf(0.001f, roughness * roughness);
		static const float kMinCosTheta = 1e-4f;

		float thetaM = atanf((alpha * sqrtf(u1)) / sqrt(1.f - u1));
		float phiM = (2.f * static_cast<float>(CUDART_PI) * u2);

		vec3 m = TransformToWorld(sinf(thetaM) * cosf(phiM), sinf(thetaM) * sinf(phiM), cosf(thetaM), res.HitNormal);
		m.make_unit_vector();

		if (dot(wo, m) < kMinCosTheta) {
			return false;
		}

		targetRay.origin = res.HitPos + (res.HitNormal * 0.01f);
		targetRay.direction = (2.f * PROTECTZERO(fabsf(dot(wo, m))) * m - wo);

		// pdf = D(m)|m * n|

		
		pdf = GGXDistribution(alpha, thetaM, res.HitNormal, m) * fabsf(dot(m, res.HitNormal));
			
		// evaluate cook-torrance

		vec3 hr = sign(dot(wi, res.HitNormal)) * (wi + wo);
		hr.make_unit_vector();

		float f = fabsf((1.f - target->lighting.ior) / (1.f + target->lighting.ior));
		// in my schlick's function, f0 is represented as r0
		float F0 = (f * f);
		vec3 finalSchlicksInput = lerpVectors(vec3(F0), res.HitAlbedo, metalness);
		vec3 fresnelTerm = coloredSchlick(finalSchlicksInput, dot(wi, m), target->lighting.ior);


		vec3 numerator = fresnelTerm * GGXDistribution(alpha, thetaM, res.HitNormal, m) * GGXGeometry(wi, wo, m, res.HitNormal, alpha);
		// float denominator = 4.f * (dot(wi, res.HitNormal)) * (dot(wo, res.HitNormal));
		
		attenuation = numerator;


		

		//Eval(alpha, metalness, target, res.HitNormal, wo, wi, res.HitAlbedo, attenuation, pdf);
		// frensel term being wacky..
		return true;
	}

	__device__ float PDF(const HitResult& res, Object* target, const vec3& wo, const vec3& wi) {
		float alpha = target->lighting.roughness;
		static const float kMinCosTheta = 1e-6f;

		if (target->pbrMaps.mraoMap.initialized) {
			alpha = res.MRAO.b();
		}

		if (min(wo.z(), wi.z()) < kMinCosTheta)
			return 0.f;

		vec3 h = unit_vector(wo + wi);
		float woDotH = dot(wo, h);

		float pdf = FalcorNDFGGX(alpha, h.z());
	    // return GGXDistribution(alpha, h.z(), res.HitNormal, h) * fabsf(dot(h, res.HitNormal));

		return pdf / (4.0 * woDotH);
	}
}


