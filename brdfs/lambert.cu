﻿#include "lambert.cuh"
#include "../vec3.cuh"
#include "../ray.cuh"
#include "../hitresult.cuh"
#include "../object.cuh"

#include "curand_kernel.h"
#include "math.h"
#include "math_constants.h"

using namespace Tracer;

#define RANDVEC3 vec3(fmodf(curand_uniform(local_rand_state) + extraRand, 1.f),fmodf(curand_uniform(local_rand_state) + extraRand, 1.f),fmodf(curand_uniform(local_rand_state) + extraRand, 1.f))
//#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

static constexpr float M_1_PI = 0.318309886183790671538f;

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


namespace Tracer {
	namespace LambertBRDF {
		__device__ vec3 random_in_unit_sphere(curandState* local_rand_state, float extraRand) {
			vec3 p;
		
			do {
				p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
			} while (p.squared_length() >= 1.0f);
			return p;
		}
		
		__device__ void SampleWorld(const HitResult& rec, curandState* local_rand_state, float extraRand, vec3& attenuation, Ray& targetRay, Object* target) {
			/*vec3 newDirPos = rec.HitPos + rec.HitNormal + random_in_unit_sphere(local_rand_state, extraRand);
			targetRay.origin = rec.HitPos;
			targetRay.direction = unit_vector(newDirPos - rec.HitPos);
			*/

			targetRay.origin = rec.HitPos;

			float r1 = curand_uniform(local_rand_state);
			float r2 = curand_uniform(local_rand_state);

			float r = sqrtf(r1);
			float theta = r2 * 2.f * CUDART_PI;

			float x = r * cosf(theta);
			float y = r * sinf(theta);

			// Project z up to the unit hemisphere
			float z = sqrt(1.0f - x * x - y * y);

			vec3 sampleLocalized = TransformToWorld(x, y, z, rec.HitNormal);

			float cos_theta = dot(sampleLocalized, rec.HitNormal);

			targetRay.direction = sampleLocalized;

			attenuation = ((target->color * target->emission) * cos_theta);

		}
	}
}