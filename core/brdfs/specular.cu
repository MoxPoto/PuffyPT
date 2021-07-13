﻿#include <brdfs/specular.cuh>
#include <brdfs/lambert.cuh>

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

#include "curand_kernel.h"
#include "math_constants.h"


#define PROTECTZERO(statement) fmaxf(0.001f, statement)

// from: https://computergraphics.stackexchange.com/questions/4979/what-is-importance-sampling/4993
__device__ static vec3 TransformToWorld(const float& x, const float& y, const float& z, const vec3& normal)
{
	// Find an axis that is not parallel to normal
	vec3 majorAxis;
	if (fabsf(normal.x()) < 0.57735026919f /* 1 / sqrt(3) */) {
		majorAxis = vec3(1, 0, 0);
	}
	else if (fabsf(normal.y()) < 0.57735026919f /* 1 / sqrt(3) */) {
		majorAxis = vec3(0, 1, 0);
	}
	else {
		majorAxis = vec3(0, 0, 1);
	}

	// Use majorAxis to create a coordinate system relative to world space
	vec3 u = unit_vector(cross(normal, majorAxis));
	vec3 v = cross(normal, u);
	vec3 w = normal;

	// Transform from local coordinates to world coordinates
	return u * x + v * y + w * z;
}

__device__ static inline float lerp(float a, float b, float f)
{
	return (a * (1.0f - f)) + (b * f);
}

__device__ static inline vec3 lerpVectors(vec3 a, vec3 b, float f)
{
	return (a * (1.0f - f)) + (b * f);
}

__device__ static float thetaFromVec(vec3 vec) {
	return atanf(sqrtf(vec.x() * vec.x() + vec.y() * vec.y()) / vec.z());
}

// χ+(a)
__device__ static float chi(float num) {
	return num >= 0.f ? 1.f : 0.f;
}

// D(m)
__device__ static float GGXDistribution(float width, float thetaM, float phiM, const vec3& hitNormal, const vec3& microfacet) {
	float alphaSquared = (width * width);

	float numerator = alphaSquared * chi(dot(microfacet, hitNormal));
	float tanThetaM = tanf(thetaM);
	float term2 = (alphaSquared + powf(tanThetaM, 2));

	float denominator = (static_cast<float>(CUDART_PI) * powf(cosf(thetaM), 4.f) * (term2 * term2));

	return numerator / denominator;
}

// G1(v, m)
__device__ static float GGXGeometry(const vec3& v, const vec3& n, const vec3& m, float width) {
	float vdotm = dot(v, m);
	float vdotn = dot(v, n);

	float chiOfDot = chi(vdotm / vdotn);
	float alphaSquared = (width * width);
	float thetaOfV = thetaFromVec(v);
	float tanPart = tanf(thetaOfV);
	float denominator = 1.f + sqrtf(1.f + (alphaSquared * (tanPart * tanPart)));

	return chiOfDot * (2.f / denominator);
}


__device__ static vec3 coloredSchlick(vec3 r0, float cosine, float ref_idx) {
	return r0 + (vec3(1.0f) - r0) * powf((1.0 - cosine), 2.0f);
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

	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, float& pdf, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target) {
		// FIXME: when some rays are at a specific direction, the color returned is pure black.. most likely some dividing by zero issue.. maybe
		
		// wo = -previousRay.direction;
		// wi = reflect(-wo, hitnormal);

		// some of the code is based on: https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/shaders.cpp#L63

		vec3 wo = -previousRay.direction;
		vec3 wi = reflect(previousRay.direction, res.HitNormal);
		float u1 = curand_uniform(local_rand_state);
		float u2 = curand_uniform(local_rand_state);
		// random 1 and 2 in the cook-torrance paper
		float alpha = fmaxf(0.001f, target->lighting.roughness * target->lighting.roughness);

		float thetaM = atanf((alpha * sqrtf(u1)) / sqrt(1.f - u1));
		float phiM = (2.f * static_cast<float>(CUDART_PI) * u2);

		vec3 m = TransformToWorld(sinf(thetaM) * cosf(phiM), sinf(thetaM) * sinf(phiM), cosf(thetaM), res.HitNormal);
		m.make_unit_vector();

		if (dot(m, wo) < 0.f) {
			m = reflect(-m, res.HitNormal);
		}

		targetRay.origin = res.HitPos + (res.HitNormal * 0.01f);
		targetRay.direction = (2.f * PROTECTZERO(fabsf(dot(wo, m))) * m - wo);

		pdf = GGXDistribution(alpha, thetaM, phiM, res.HitNormal, m) * fabsf((dot(m, res.HitNormal))) / (4.f * fabsf((dot(targetRay.direction, m))));
			
		// evaluate cook-torrance

		float f = (target->lighting.ior - 1.f) / (target->lighting.ior + 1.f);
		// in my schlick's function, f0 is represented as r0
		float F0 = (f * f);

		vec3 finalSchlicksInput = lerpVectors(vec3(F0), target->GetColor(res), target->lighting.metalness);
		vec3 fresnelTerm = coloredSchlick(finalSchlicksInput, dot(wo, m), target->lighting.ior);

		vec3 numerator = GGXDistribution(alpha, thetaM, phiM, res.HitNormal, m) * fresnelTerm * GGXGeometry(targetRay.direction, res.HitNormal, m, alpha);
		float denominator = 4.f * (dot(res.HitNormal, wi) * dot(res.HitNormal, wo));

		attenuation = numerator / denominator;

		// frensel term being wacky..
	}
}


