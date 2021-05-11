#include "mainDenoiser.cuh"
#include "../vec3.cuh"
#include "cuda_runtime.h"
#include "math.h"
#include "../dxhook/mainHook.h"
#include "device_launch_parameters.h"

#define GBUFFER_AT(x, y) gbufferData[y * width + x];
#define DIFFUSE_FILTER 3

namespace Tracer {
	namespace Denoising {
		__global__ void denoise(GBuffer** gbufferData, float* framebuffer, int width, int height) {

			int i = threadIdx.x + blockIdx.x * blockDim.x;
			int j = threadIdx.y + blockIdx.y * blockDim.y;
			if ((i >= width) || (j >= height)) return;
			int pixel_index = j * width * 3 + i * 3;
			int gbuffer_idx = j * width + i;

			GBuffer* ourBuffer = GBUFFER_AT(i, j);
			vec3 denoisedResult = ourBuffer->diffuse;
			int passes = 1;

			for (int fX = i - DIFFUSE_FILTER; fX < (i + DIFFUSE_FILTER); fX++) {
				for (int fY = j - DIFFUSE_FILTER; fY < (j + DIFFUSE_FILTER); fY++) {
					if ((fX >= 0 && fX < width) && (fY >= 0 && fY < height)) {
						GBuffer* gBufferThere = GBUFFER_AT(fX, fY);
						
						denoisedResult += gBufferThere->diffuse;
						passes++;
					}
				}
			}

			denoisedResult /= (float)passes;

			framebuffer[pixel_index + 0] = denoisedResult.r();
			framebuffer[pixel_index + 1] = denoisedResult.g();
			framebuffer[pixel_index + 2] = denoisedResult.b();

		}
	}
}