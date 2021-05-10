#include "lambert.cuh"
#include "../vec3.cuh"
#include "../ray.cuh"
#include "../hitresult.cuh"
#include "../object.cuh"

#include "curand_kernel.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

namespace Tracer {
	namespace LambertBRDF {
		__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
			vec3 p;
			do {
				p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
			} while (p.squared_length() >= 1.0f);
			return p;
		}
		
		__device__ void SampleWorld(const HitResult& rec, curandState* local_rand_state, vec3& attenuation, Ray& targetRay, Object* target) {
			vec3 newDirPos = rec.HitPos + rec.HitNormal + random_in_unit_sphere(local_rand_state);
			targetRay.origin = rec.HitPos;
			targetRay.direction = unit_vector(newDirPos - rec.HitPos);

			attenuation = ((target->color * target->emission));

		}
	}
}