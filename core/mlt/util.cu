#include <mlt/util.cuh>
#include <classes/vec3.cuh>

#include "math_constants.h"

namespace MLT {
	vec3 PerturbVector(const vec3& dir, float randomVariable, float randomVariable2) {
		const float theta1 = 0.0001f;
		const float theta2 = 0.1f;

		const vec3 cross1(1.f, 0.f, 0.f);
		const vec3 cross2(0.f, 1.f, 0.f);
		// Suggested parameters by Veach and Guibas

		vec3 U, V;

		if (fabsf(dir.x()) < 0.5f) {
			U = cross(dir, cross1);
		}
		else {
			U = cross(dir, cross2);
		}

		U.make_unit_vector();

		V = cross(U, dir);

		// determine the offsets
		float phi = randomVariable * 2.f * static_cast<float>(CUDART_PI);
		float r = theta2 * expf(-logf(theta2 / theta1) * randomVariable2);

		vec3 newDirection = dir;
		newDirection += r * cosf(phi) * U + r * sinf(phi) * V;
		newDirection.make_unit_vector();
		
		return newDirection;
	}

	void PixelOffset(float r1, float r2, float& x, float& y, float randVariable, float randVariable2) {
		float phi = randVariable * 2.f * static_cast<float>(CUDART_PI);
		float r = r2 * expf(-logf(r2 / r1) * randVariable2);

		x += r * cosf(phi);
		y += r * sinf(phi);
	}
}