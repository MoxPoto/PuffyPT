#include "specular.cuh"
#include "lambert.cuh"

#include "../vec3.cuh"
#include "../ray.cuh"
#include "../hitresult.cuh"
#include "../object.cuh"

#include "curand_kernel.h"

namespace Tracer {
	namespace SpecularBRDF {
		__device__ vec3 reflect(const vec3& direction, const vec3& normal) {
			return direction - 2.0f * dot(direction, normal) * normal;
		}

		__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target) {
			targetRay.origin = res.HitPos;
			targetRay.direction = reflect(previousRay.direction, res.HitNormal);

			attenuation = (target->color * target->emission);

			if (target->lighting.roughness > 0.05f) {
				vec3 sampleDir = target->lighting.roughness * LambertBRDF::random_in_unit_sphere(local_rand_state);

				targetRay.direction = unit_vector(targetRay.direction + sampleDir);
			}
		}
	}
}

