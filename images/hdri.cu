#include "hdri.cuh"
#include "../vec3.cuh"
#include "math_constants.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include "../util/macros.h"

// can't really include stb_image in a cuda file (device code specifically)
// so i'm jerryrigging some typedefs from the original file
typedef float stbi_uc;

static inline float sign(const float& value) {
	if (value < 0.f) return -1.f;

	return 1.f;
}

namespace Tracer {
	__host__ __device__ HDRI::HDRI() {
		resX = 0;
		resY = 0; // this is such an awesome constructor

	}


	__host__ __device__ vec3 HDRI::getPixel(const int& x, const int& y, float* imagePtr) {
		//if (imagePtr != nullptr) {
			int base_index = (3 * (y * resX + x));

			return vec3(imagePtr[base_index], imagePtr[base_index + 1], imagePtr[base_index + 2]);
		//}

		//return vec3(1, 0, 0);
	}

	__host__ __device__ float HDRI::getPitch(const vec3& dir) { // https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/pathtracer.cpp#L97
		if (dir.x() == 0 && dir.y() == 0) return CUDART_PI / 2.f * sign(dir.z());
		return asinf(abs(dir.z())) * sign(dir.z());
	}

	__host__ __device__ float HDRI::getYaw(const vec3& N) { // https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/pathtracer.cpp#L103
		if (N.y() == 0) return N.x() >= 0 ? 0 : CUDART_PI;
		vec3 base = unit_vector((N * vec3(1, 1, 0)));
		return acosf(dot(base, vec3(1, 0, 0))) * sign(N.y());
	}

	__host__ __device__ vec3 HDRI::getPixelFromRay(const vec3& N, float* imagePtr) {
		// if (imagePtr == nullptr) return vec3(0, 1, 0);

		float y = resY / 2.f - ((getPitch(N) * 2.f) / CUDART_PI) * resY / 2.f;
		float x = resX / 2.f + (getYaw(N) / CUDART_PI) * resX / 2.f;
		
		y = (fmodf(floorf(y), resY - 1.f));
		x = (fmodf(floorf(x), resX - 1.f));

		
		vec3 color = getPixel(x, y, imagePtr);
		color = color * 0.1f;

		return color;
	}
}