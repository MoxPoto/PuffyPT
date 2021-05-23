#include "hdri.cuh"
#include "../vec3.cuh"
#include "math_constants.h"

// can't really include stb_image in a cuda file (device code specifically)
// so i'm jerryrigging some typedefs from the original file
typedef unsigned char stbi_uc;

static inline float sign(const float& value) {
	if (value < 0.f) return -1.f;

	return 1.f;
}

namespace Tracer {
	__host__ __device__ HDRI::HDRI() {
		resX = 0;
		resY = 0; // this is such an awesome constructor
	}

	__host__ __device__ void HDRI::loadData(unsigned char* newData) {
		imageData = newData;
	}

	__host__ __device__ vec3 HDRI::getPixel(const int& x, const int& y) {
		if (imageData != nullptr) {
			const stbi_uc* pixel = imageData + (3 * (y * resX + x));

			return vec3(pixel[0] / 255.0f, pixel[1] / 255.0f, pixel[2] / 255.0f);
		}

		return vec3(1, 0, 0);
	}

	__host__ __device__ float HDRI::getPitch(const vec3& dir) { // https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/pathtracer.cpp#L97
		if (dir.x() == 0 && dir.y() == 0) return CUDART_PI / 2.f * sign(dir.z());
		return asinf(abs(dir.z())) * sign(dir.z());
	}

	__host__ __device__ float HDRI::getYaw(const vec3& N) { // https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/pathtracer.cpp#L103
		if (N.y() == 0) return N.x() >= 0 ? 0 : CUDART_PI;
		return acosf(dot(N * (unit_vector(vec3(1.f, 1.f, 0))), vec3(1.f, 0, 0))) * sign(N.y);
	}

	__host__ __device__ vec3 HDRI::getPixelFromRay(const vec3& N) {
		if (imageData == NULL) return vec3(0, 0, 0);

		double y = resY / 2.f - ((getPitch(N) * 2.f) / CUDART_PI) * resY / 2.f;
		double x = resX / 2.f + (getYaw(N) / CUDART_PI) * resX / 2.f;

		y = (fmodf(floorf(y), resY - 1.f));
		x = (fmodf(floorf(x), resX - 1.f));

		// std::cout << std::to_string(y) << " : " << std::to_string(x) << "\nWith the dir: " << vectorAsAString(N) << std::endl;

		vec3 color = getPixel(x, y);

		color = color * (color * 1);

		return color;
	}
}