#include <postprocess/mainDenoiser.cuh>
#include <classes/vec3.cuh>
#include "cuda_runtime.h"
#include "math.h"
#include <dxhook/mainHook.h>
#include "device_launch_parameters.h"
#include <classes/object.cuh>

#define GBUFFER_AT(x, y) (gbufferData + (y * width + x));



namespace Post {
	__global__ void denoise(GBuffer* gbufferData, float* realFB, float* framebuffer, int width, int height) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int pixel_index = j * width * 3 + i * 3;
		int gbuffer_idx = j * width + i;

		GBuffer* ourBuffer = GBUFFER_AT(i, j);
		vec3 denoisedResult = ourBuffer->diffuse;

		int FILTER_SIZE = 11;

		vec3 blurredArea(0, 0, 0);
		int passes = 0;

		float accumulatedLuminance = 0.f;
		int lumPasses = 0;

		for (int fX = i - FILTER_SIZE; fX <= i + FILTER_SIZE; fX++) {
			for (int fY = j - FILTER_SIZE; fY <= j + FILTER_SIZE; fY++) {
				if (!(fX > 0 && fX < width && fY > 0 && fY < height))
					continue;

				GBuffer* thisBuffer = GBUFFER_AT(fX, fY);
				if (!(ourBuffer->normal == thisBuffer->normal) && fabsf(ourBuffer->depth - thisBuffer->depth) > 5)
					continue;


				accumulatedLuminance += luminance(thisBuffer->diffuse);
				lumPasses++;
				

				blurredArea += thisBuffer->diffuse;
				passes++;
			}
		}

		if (passes > 0) {
			accumulatedLuminance /= lumPasses;
			blurredArea /= passes;

			float thisLuminance = luminance(ourBuffer->diffuse);


			denoisedResult = blurredArea * fabsf((thisLuminance - accumulatedLuminance));
		}

		framebuffer[pixel_index + 0] = denoisedResult.r();
		framebuffer[pixel_index + 1] = denoisedResult.g();
		framebuffer[pixel_index + 2] = denoisedResult.b();

	}
}
