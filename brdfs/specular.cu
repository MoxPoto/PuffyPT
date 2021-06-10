#include "specular.cuh"
#include "lambert.cuh"

#include "../vec3.cuh"
#include "../ray.cuh"
#include "../hitresult.cuh"
#include "../object.cuh"

#include "curand_kernel.h"

__device__ static inline float lerp(float a, float b, float f)
{
	return (a * (1.0f - f)) + (b * f);
}

namespace Tracer {
	namespace SpecularBRDF {
		__device__ vec3 reflect(const vec3& direction, const vec3& normal) {
			return direction - 2.0f * dot(direction, normal) * normal;
		}

		__device__ float schlick(float cosine, float ref_idx) {
			float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
			r0 = r0 * r0;
			return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
		}

		__device__ void SampleWorld(const HitResult& res, curandState* local_rand_state, float extraRand, const Ray& previousRay, vec3& attenuation, Ray& targetRay, Object* target) {
			targetRay.origin = res.HitPos + (res.HitNormal * 0.001f);
			targetRay.direction = reflect(previousRay.direction, res.HitNormal);

			attenuation = (target->color * target->emission);

			if (target->lighting.roughness > 0.03f) {
				float fresnelApprox = schlick(dot(-previousRay.direction, res.HitNormal), target->lighting.ior);
				fresnelApprox = lerp(fresnelApprox, 0.f, 1.f - target->lighting.roughness * target->lighting.roughness); // Weight fresnel approximation by the roughness to pure specular

				vec3 sampleDir = fresnelApprox * LambertBRDF::random_in_unit_sphere(local_rand_state, extraRand);

				targetRay.direction = unit_vector(targetRay.direction + sampleDir);
			}
		}
	}
}

