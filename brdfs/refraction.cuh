#include "../vec3.cuh"
#include "../hitresult.cuh"
#include "curand_kernel.h"
#include "../ray.cuh"
#include "../object.cuh"

namespace Tracer {
	namespace RefractBRDF {
		__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float& pdf, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target);
	}
}