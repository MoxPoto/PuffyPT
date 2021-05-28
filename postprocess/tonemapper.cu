#include "mainDenoiser.cuh"
#include "../vec3.cuh"
#include "cuda_runtime.h"
#include "math.h"
#include "../dxhook/mainHook.h"
#include "device_launch_parameters.h"
#include "../object.cuh"

#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

static constexpr float POW_ARG = 1.0f / 2.4f;

namespace Tracer {
	namespace Post {
		__device__ vec3 LinearTosRGB(vec3 color)
		{
			vec3 x = color * 12.92f;
			vec3 clampedCol = color;
			clampedCol.clamp();

			vec3 y = 1.055f * vec3(powf(clampedCol.r(), POW_ARG), powf(clampedCol.g(), POW_ARG), powf(clampedCol.b(), POW_ARG)) - vec3(0.055f, 0.055f, 0.055f);

			
			float newR = color.r() < 0.0031308f ? x.r() : y.r();
			float newG = color.g() < 0.0031308f ? x.g() : y.g();
			float newB = color.b() < 0.0031308f ? x.b() : y.b();

			vec3 clr(newR, newG, newB);
			return clr;
		}

		__device__ float luminance(vec3 rgb) {
			// gets the brightness of a rgb pixel using weighted rgb contribution vector

			return dot(rgb, vec3(0.2126f, 0.7152f, 0.0722f));
		}

		__global__ void tonemap(float* framebuffer, Camera mainCam, float* postFB, int width, int height) {
			/*
			ACES Approximation by Krzysztof Narkowicz
			https://64.github.io/tonemapping/#aces
			*/
			int i = threadIdx.x + blockIdx.x * blockDim.x;
			int j = threadIdx.y + blockIdx.y * blockDim.y;
			if ((i >= width) || (j >= height)) return;
			int pixel_index = j * width * 3 + i * 3;

			vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]) * mainCam.exposure;

			frameColor *= 0.5f;
			float a = 2.51f;
			float b = 0.03f;
			float c = 2.43f;
			float d = 0.59f;
			float e = 0.14f;
			vec3 tonemapped = (frameColor * (a * frameColor + vec3(b, b, b))) / (frameColor * (c * frameColor + vec3(d, d, d)) + vec3(e, e, e));

			tonemapped.clamp();

			tonemapped = LinearTosRGB(tonemapped);

			postFB[pixel_index] = tonemapped.r();
			postFB[pixel_index + 1] = tonemapped.g();
			postFB[pixel_index + 2] = tonemapped.b();
		}
	}

	__host__ void ApplyPostprocess(int width, int height, dim3 blocks, dim3 threads) {
		using namespace Post;

		tonemap << <blocks, threads >> > (DXHook::fb, DXHook::mainCam, DXHook::postFB, width, height);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}