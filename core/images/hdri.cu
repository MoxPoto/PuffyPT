#include <images/hdri.cuh>
#include <classes/vec3.cuh>
#include <util/macros.h>

#include <math/basic.cuh>
#include "math_constants.h"
#include "cuda_runtime.h"
#include <stdio.h>
// can't really include stb_image in a cuda file (device code specifically)
// so i'm jerryrigging some typedefs from the original file
typedef float stbi_uc;
static constexpr float M_2_PI = 0.636619772367581343076f;

__device__ HDRI::HDRI() {
	resX = 0;
	resY = 0; // this is such an awesome constructor
	brightness = 1.f;
}

__device__ vec3 HDRI::getPixel(const int& x, const int& y, float* imagePtr) {
	//if (imagePtr != nullptr) {
		int base_index = (3 * (fmaf(y, resX, x)));
		// fmaf is a cuda intrinistic

		return vec3(imagePtr[base_index], imagePtr[base_index + 1], imagePtr[base_index + 2]);
	//}

	//return vec3(1, 0, 0);
}

__device__ float HDRI::getPitch(const vec3& dir) { // https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/pathtracer.cpp#L97
	if (dir.x() == 0 && dir.y() == 0) return CUDART_PI / 2.f * sign(dir.z());
	return asinf(abs(dir.z())) * sign(dir.z());
}

__device__ float HDRI::getYaw(const vec3& N) { // https://github.com/100PXSquared/gmod-binary-tracer/blob/56f482c041909494497d22dcf5c45d4f507aa022/Binary%20Module/pathtracer.cpp#L103
	if (N.y() == 0) return N.x() >= 0 ? 0 : CUDART_PI;
	vec3 base = unit_vector((N * vec3(1, 1, 0)));
	return acosf(dot(base, vec3(1, 0, 0))) * sign(N.y());
}

__device__ vec3 HDRI::GetPixelFromRay(const vec3& N, float* imagePtr) {
	// if (imagePtr == nullptr) return vec3(0, 1, 0);
	const float deg2rad = CUDART_PI / 180.f;

	float y = (1.f - fmodf(getPitch(N) + pitch * deg2rad, CUDART_PI * 2) * M_2_PI) * resY / 2.f;
	float x = resX - (1.f + fmodf(getYaw(N) + yaw * deg2rad, CUDART_PI * 2) / CUDART_PI) * resX / 2.f;
		
	x -= resX * floorf(x / resX);
	y -= resY * floorf(y / resY);
		
	vec3 color = getPixel(floorf(x), floorf(y), imagePtr);
	color *= brightness;

	return color;
}
