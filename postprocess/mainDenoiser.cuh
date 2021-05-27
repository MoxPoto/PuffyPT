#ifndef MDENOISER_H
#define MDENOISER_H

#include "../vec3.cuh"
#include "../object.cuh"
#include "cuda_runtime.h"

// originally this was denoising only

namespace Tracer {
	namespace Post {
		struct GBuffer {
			vec3 normal = vec3(0, 0, 0);
			vec3 position = vec3(0, 0, 0);
			int objectID = 0;
			float depth = 0.f;
			vec3 albedo = vec3(0, 0, 0);
			vec3 diffuse = vec3(0, 0, 0);
			bool isSky = false;
			BRDF brdfType = BRDF::Lambertian;
		};

		__global__ extern void denoise(GBuffer* gbufferData, float* framebuffer, int width, int height);
		__global__ extern void tonemap(float* framebuffer, float* postFB, int width, int height);

		__device__ extern float luminance(vec3 rgb);
		__device__ vec3 LinearTosRGB(vec3 color);
	}

	__host__ extern void ApplyPostprocess(int width, int height, dim3 blocks, dim3 threads);
}
#endif