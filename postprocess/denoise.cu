#include "mainDenoiser.cuh"
#include "../vec3.cuh"
#include "cuda_runtime.h"
#include "math.h"
#include "../dxhook/mainHook.h"
#include "device_launch_parameters.h"
#include "../object.cuh"

#define GBUFFER_AT(x, y) (gbufferData + (y * width + x));

namespace Tracer {
	namespace Post {
		__global__ void denoise(GBuffer* gbufferData, float* framebuffer, int width, int height) {

			int i = threadIdx.x + blockIdx.x * blockDim.x;
			int j = threadIdx.y + blockIdx.y * blockDim.y;
			if ((i >= width) || (j >= height)) return;
			int pixel_index = j * width * 3 + i * 3;
			int gbuffer_idx = j * width + i;

			GBuffer* ourBuffer = GBUFFER_AT(i, j);
			
			if (ourBuffer->isSky) {
				return;
			}

			vec3 denoisedResult = ourBuffer->diffuse;
			int passes = 1;
			float brightness = 0.0f;
			int DIFFUSE_FILTER = 7;
			float brightness_factor = 0.006f;

			if (ourBuffer->brdfType == BRDF::Specular) {
				DIFFUSE_FILTER = 4; // Lower filtering for specular surfaces
				brightness_factor = 0.04f;
			}

				for (int fX = i - DIFFUSE_FILTER; fX < (i + DIFFUSE_FILTER); fX++) {
					for (int fY = j - DIFFUSE_FILTER; fY < (j + DIFFUSE_FILTER); fY++) {
						if ((fX >= 0 && fX < width) && (fY >= 0 && fY < height)) {
							GBuffer* gBufferThere = GBUFFER_AT(fX, fY);

							if (gBufferThere->objectID == ourBuffer->objectID && !gBufferThere->isSky) {
								denoisedResult += gBufferThere->diffuse;
								brightness += (gBufferThere->diffuse.r() + gBufferThere->diffuse.g() + gBufferThere->diffuse.b()) / 3.f;
								passes++;
							}
						}
					}
				}

			denoisedResult /= (float)passes;
			denoisedResult = (ourBuffer->albedo + (denoisedResult * (brightness_factor * brightness))) / 2.f;
			denoisedResult.clamp();

			framebuffer[pixel_index + 0] = denoisedResult.r();
			framebuffer[pixel_index + 1] = denoisedResult.g();
			framebuffer[pixel_index + 2] = denoisedResult.b();

		}
	}
}