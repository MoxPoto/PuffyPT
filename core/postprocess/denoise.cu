#include <postprocess/mainDenoiser.cuh>
#include <classes/vec3.cuh>
#include "cuda_runtime.h"
#include "math.h"
#include <dxhook/mainHook.h>
#include "device_launch_parameters.h"
#include <classes/object.cuh>

#define GBUFFER_AT(x, y) (gbufferData + (y * width + x));

static float c_phi = 1.f;
static float n_phi = 1.f;
static float p_phi = 1.f;
static float stepwidth = 2.f;


namespace Post {
	__global__ void denoise(GBuffer* gbufferData, float* realFB, float* framebuffer, int width, int height) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int pixel_index = j * width * 3 + i * 3;
		int gbuffer_idx = j * width + i;

		GBuffer* ourBuffer = GBUFFER_AT(i, j);
		vec3 denoisedResult(0, 0, 0);



		framebuffer[pixel_index + 0] = denoisedResult.r();
		framebuffer[pixel_index + 1] = denoisedResult.g();
		framebuffer[pixel_index + 2] = denoisedResult.b();

	}
}
