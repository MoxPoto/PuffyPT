#include <postprocess/mainDenoiser.cuh>
#include <classes/vec3.cuh>
#include "cuda_runtime.h"
#include "math.h"
#include <dxhook/mainHook.h>
#include "device_launch_parameters.h"
#include <classes/object.cuh>
#include <classes/camera.cuh>

#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

static constexpr float POW_ARG = 1.0f / 2.4f;

static __global__ void blur(float* framebuffer, float* blurFB, int width, int height) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width * 3 + i * 3;


	vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]);
	const int FILTER_SIZE = 16;
	const int REAL_SIZE = 10;
	const vec3 ourPosition(i, j, 0);

	vec3 blurred(0, 0, 0);
	int passes = 0;

	for (int fX = i - FILTER_SIZE; fX <= i + FILTER_SIZE; fX++) {
		for (int fY = j - FILTER_SIZE; fY <= j + FILTER_SIZE; fY++) {
			const vec3 thisPosition(fX, fY, 0);

			if ((ourPosition - thisPosition).squared_length() > REAL_SIZE * REAL_SIZE)
				continue;

			if (fX > 0 && fX < width && fY > 0 && fY < height) {
				int pixel_index = fY * width * 3 + fX * 3;

				if (fX == i && fY == j)
					continue;
				vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]);

				if (!isnan(frameColor.r()) && !isnan(frameColor.g()) && !isnan(frameColor.b())) {
					blurred += frameColor;
					passes++;
				}
				
			}
		}
	}

	if (passes <= 0) {
		return;
	}

	blurred /= (fmaxf(passes, 1));


	blurFB[pixel_index] = blurred.r();
	blurFB[pixel_index + 1] = blurred.g();
	blurFB[pixel_index + 2] = blurred.b();

}

static __global__ void copy(float* srcBuffer, float* dstBuffer, int width, int height) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width * 3 + i * 3;

	vec3 frameColor = vec3(srcBuffer[pixel_index], srcBuffer[pixel_index + 1], srcBuffer[pixel_index + 2]);

	dstBuffer[pixel_index] = frameColor.r();
	dstBuffer[pixel_index + 1] = frameColor.g();
	dstBuffer[pixel_index + 2] = frameColor.b();

}

namespace Post {
	__device__ vec3 LinearTosRGB(vec3 color)
	{
		vec3 x = color * 12.92f;
		vec3 clampedCol = vec3(__saturatef(color.r()), __saturatef(color.g()), __saturatef(color.b()));

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

	__global__ void tonemap(float* framebuffer, Camera mainCam, float* postFB, float* bloomFB, int width, int height) {
		/*
		ACES Approximation by Krzysztof Narkowicz
		https://64.github.io/tonemapping/#aces
		*/
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int pixel_index = j * width * 3 + i * 3;

		vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]) * mainCam.exposure;
		vec3 bloomColor = vec3(bloomFB[pixel_index], bloomFB[pixel_index + 1], bloomFB[pixel_index + 2]);

		//frameColor = (frameColor + bloomColor) / 2;
			
		frameColor *= 0.5f;
		float a = 2.51f;
		float b = 0.03f;
		float c = 1.43f;
		float d = 0.59f;
		float e = 0.14f;
		vec3 tonemapped = (frameColor * (a * frameColor + vec3(b, b, b))) / (frameColor * (c * frameColor + vec3(d, d, d)) + vec3(e, e, e));

		tonemapped.clamp();

		tonemapped = LinearTosRGB(tonemapped);

		
		postFB[pixel_index] = tonemapped.r();
		postFB[pixel_index + 1] = tonemapped.g();
		postFB[pixel_index + 2] = tonemapped.b();
	}

	__global__ void bloom(float* framebuffer, float* postFB, int width, int height) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int pixel_index = j * width * 3 + i * 3;
			
		vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]);
		const int FILTER_SIZE = 12;
		const int REAL_SIZE = 10;
		const vec3 ourPosition(i, j, 0);

		const float MINIMUM = 1.4f;

		vec3 newBrightness(0, 0, 0);
		int passes = 0;

		for (int fX = i - FILTER_SIZE; fX <= i + FILTER_SIZE; fX++) {
			for (int fY = j - FILTER_SIZE; fY <= j + FILTER_SIZE; fY++) {
				const vec3 thisPosition(fX, fY, 0);

				if ((ourPosition - thisPosition).squared_length() > REAL_SIZE * REAL_SIZE)
					continue;

				if (fX > 0 && fX < width && fY > 0 && fY < height) {
					int pixel_index = fY * width * 3 + fX * 3;
					vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]);
					float brightness = luminance(frameColor);

					if (brightness > MINIMUM) {
						newBrightness += frameColor * (brightness - MINIMUM);
						passes++;
					}
				}
			}
		}

		if (passes <= 0) {
			postFB[pixel_index] = 0.f;
			postFB[pixel_index + 1] = 0.f;
			postFB[pixel_index + 2] = 0.f;
			return;
		}

		newBrightness /= (fmaxf(passes, 1));

		frameColor *= ((newBrightness));

		postFB[pixel_index] = frameColor.r();
		postFB[pixel_index + 1] = frameColor.g();
		postFB[pixel_index + 2] = frameColor.b();
			
	}

}

__global__ void ClearFramebuffer(float* framebuffer, int width, int height) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width * 3 + i * 3;

	framebuffer[pixel_index] = 0.0f;
	framebuffer[pixel_index + 1] = 0.0f;
	framebuffer[pixel_index + 2] = 0.0f;
}

__host__ void ApplyPostprocess(int width, int height, dim3 blocks, dim3 threads) {
	using namespace Post;
		
	/*
	bloom << <blocks, threads >> > (DXHook::fb, DXHook::bloomFB, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

		
	blur << <blocks, threads >> > (DXHook::bloomFB, DXHook::blurFB, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	copy << <blocks, threads >> > (DXHook::blurFB, DXHook::bloomFB, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
		

	denoise << <blocks, threads >> > (DXHook::gbufferData, DXHook::fb, DXHook::bloomFB, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	*/

	tonemap << <blocks, threads >> > (DXHook::fb, DXHook::mainCam, DXHook::postFB, DXHook::bloomFB, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
