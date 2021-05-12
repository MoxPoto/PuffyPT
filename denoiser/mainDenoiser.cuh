#ifndef GBUFFER_H
#define GBUFFER_H

#include "../vec3.cuh"
#include "../object.cuh"
#include "cuda_runtime.h"

namespace Tracer {
	namespace Denoising {
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
	}
}
#endif