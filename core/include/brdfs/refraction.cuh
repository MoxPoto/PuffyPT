#include <classes/vec3.cuh>
#include <classes/hitresult.cuh>
#include "curand_kernel.h"
#include <classes/ray.cuh>
#include <classes/object.cuh>

namespace RefractBRDF {
	__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float& pdf, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target, BRDF& brdfChosen);
}
