#include <postprocess/mainDenoiser.cuh>
#include <classes/vec3.cuh>
#include "cuda_runtime.h"
#include "math.h"
#include <dxhook/mainHook.h>
#include "device_launch_parameters.h"
#include <classes/object.cuh>

#define GBUFFER_AT(x, y) (gbufferData + (y * width + x));
#define GET_COLOR(i, j, colName) int _pixelIdx = j * width * 3 + i * 3; vec3 colName = vec3(realFB[_pixelIdx + 0], realFB[_pixelIdx + 1], realFB[_pixelIdx + 2]);

typedef DWORD D3DCOLOR;
#define CUDA_COLOR_TO_DX(r, g, b) ((((0xff) & 0xff) << 24) | (((r) & 0xff) << 16) | (((g) & 0xff) << 8) | ((b) & 0xff));

namespace Post {
	__global__ void denoise(GBuffer* gbufferData, float* realFB, DWORD* dxFB, float* framebuffer, int width, int height) {
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

		GET_COLOR(i, j, ourColor);

		int FILTER_SIZE = 3;

		vec3 blurredArea(0, 0, 0);
		int passes = 0;

		int lumPasses = 0;

		for (int fX = i - FILTER_SIZE; fX <= i + FILTER_SIZE; fX++) {
			for (int fY = j - FILTER_SIZE; fY <= j + FILTER_SIZE; fY++) {
				if (!(fX > 0 && fX < width && fY > 0 && fY < height))
					continue;

				GBuffer* thisBuffer = GBUFFER_AT(fX, fY);

				if ((thisBuffer->normal - ourBuffer->normal).length() > 0.01) {
					continue;
				}

				if (!(thisBuffer->objectID == ourBuffer->objectID) && fabsf(ourBuffer->depth - thisBuffer->depth) > 6)
					continue;

				GET_COLOR(fX, fY, thisColor);

				blurredArea += thisColor;
				passes++;
			}
		}

		if (passes > 0) {
			blurredArea /= passes;


			denoisedResult = blurredArea;
		}

		int r = static_cast<int>(denoisedResult.r() * 255.99);
		int g = static_cast<int>(denoisedResult.g() * 255.99);
		int Xb = static_cast<int>(denoisedResult.b() * 255.99);

		// gbuffer_idx is the same index we need..

		dxFB[gbuffer_idx] = CUDA_COLOR_TO_DX(r, g, Xb);
		framebuffer[pixel_index + 0] = denoisedResult.r();
		framebuffer[pixel_index + 1] = denoisedResult.g();
		framebuffer[pixel_index + 2] = denoisedResult.b();

	}
}
