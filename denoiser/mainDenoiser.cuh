#ifndef GBUFFER_H
#define GBUFFER_H

#include "../vec3.cuh"
#include "cuda_runtime.h"

namespace Tracer {
	namespace Denoising {
		struct GBuffer {
			vec3 normal;
			vec3 position;
			int objectID;
			float depth;
			vec3 albedo;
			vec3 diffuse;
			bool isSky;
		};

		__global__ extern void denoise(GBuffer** gbufferData, float* framebuffer, int width, int height);
	}
}
#endif