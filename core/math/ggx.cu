// GGX Definitions
#include "cuda_runtime.h"
#include <classes/vec3.cuh>
#include <math/ggx.cuh>

#include "math_constants.h"

__device__ float thetaFromVec(vec3 vec) {
	return atanf(sqrtf(vec.x() * vec.x() + vec.y() * vec.y()) / vec.z());
}

// χ+(a)
__device__ float chi(float num) {
	return num > 0.f ? 1.f : 0.f;
}

// D(m)
__device__ float GGXDistribution(float width, float thetaM, const vec3& hitNormal, const vec3& microfacet) {
	float alphaSquared = (width * width);

	float numerator = alphaSquared * chi(dot(microfacet, hitNormal));
	float tanThetaM = tanf(thetaM);
	float term2 = (alphaSquared + powf(tanThetaM, 2));

	float denominator = (static_cast<float>(CUDART_PI) * powf(cosf(thetaM), 4.f) * (term2 * term2));

	return numerator / denominator;
}

// G1(v, m)
__device__ static float GGXMonoGeometry(const vec3& v, const vec3& n, const vec3& m, float width) {
	float vdotm = dot(v, m);
	float vdotn = dot(v, n);

	float chiOfDot = chi(vdotm / vdotn);
	float alphaSquared = (width * width);
	float thetaOfV = thetaFromVec(v);
	float tanPart = tanf(thetaOfV);
	float denominator = 1.f + sqrtf(1.f + (alphaSquared * (tanPart * tanPart)));

	return chiOfDot * (2.f / denominator);
}

// G(i, o, m) 
// Approximation using smith G
__device__ float GGXGeometry(const vec3& i, const vec3& o, const vec3& m, const vec3& n, float width) {
	// G ~= G1(i, m) * G1(o, m)
	// G1(o, m) produces a black streak that gets noticeable, images look the same
	// aka, this is a bandaid-fix
	return GGXMonoGeometry(i, n, m, width);// *GGXMonoGeometry(o, n, m, width);
}