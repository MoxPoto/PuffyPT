#ifndef BXDF_CUH
#define BXDF_CUH

#include <classes/vec3.cuh>
#include <classes/hitresult.cuh>
#include "curand_kernel.h"
#include <classes/ray.cuh>
#include <classes/lighting.cuh>

struct Evaluation {
	vec3 attenuation;
	float pdf = 0.f;
};

class BxDF {
public:
	BxDF() {};

	__device__ virtual void Evaluate(const HitResult& result, const PBRMap& pbrMaps, const LightingOptions& lighting, const Ray& previousRay, Ray& targetRay, curandState* rand_state, Evaluation& eval, BRDF& brdfChosen) = 0;
};

#endif