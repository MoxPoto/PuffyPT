#include <postprocess/mainDenoiser.cuh>
#include <classes/vec3.cuh>
#include "cuda_runtime.h"
#include "math.h"
#include <dxhook/mainHook.h>
#include "device_launch_parameters.h"
#include <classes/object.cuh>
#include <classes/camera.cuh>

#include <glm/mat3x3.hpp>
#include <glm/gtx/matrix_operation.hpp>
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

static constexpr float POW_ARG = 1.0f / 2.4f;



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

// WHITE BALANCE
// Much of this is from Falcor: https://github.com/NVIDIAGameWorks/Falcor/blob/5236495554f57a734cc815522d95ae9a7dfe458a/Source/Falcor/Utils/Color/ColorUtils.h
__device__ static glm::mat3x3 kColorTransform_XYZtoLMS_CAT02;

__device__ static glm::mat3x3 kColorTransform_RGBtoXYZ_Rec709;

__device__ static glm::mat3x3 kColorTransform_XYZtoRGB_Rec709;

__device__ static glm::mat3x3 kColorTransform_LMStoXYZ_CAT02;

__device__ static vec3 xyYtoXYZ(float x, float y, float Y)
{
	return vec3(x * Y / y, Y, (1.f - x - y) * Y / y);
}

__device__ static vec3 colorTemperatureToXYZ(float T, float Y = 1.f)
{
	if (T < 1667.f || T > 25000.f)
	{
		return vec3(0, 0, 0);
	}

	// We do the computations in double
	double t = T;
	double t2 = t * t;
	double t3 = t * t * t;

	double xc = 0.0;
	if (T < 4000.f)
	{
		xc = -0.2661239e9 / t3 - 0.2343580e6 / t2 + 0.8776956e3 / t + 0.179910;
	}
	else
	{
		xc = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390;
	}

	double x = xc;
	double x2 = x * x;
	double x3 = x * x * x;

	double yc = 0.0;
	if (T < 2222.f)
	{
		yc = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
	}
	else if (T < 4000.f)
	{
		yc = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
	}
	else
	{
		yc = +3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
	}

	// Return as XYZ color.
	return xyYtoXYZ((float)xc, (float)yc, Y);
}

__device__ static glm::mat3x3 calculateWhiteBalanceTransformRGB_Rec709(float T)
{
	const glm::mat3x3 MA = kColorTransform_XYZtoLMS_CAT02 * kColorTransform_RGBtoXYZ_Rec709;    // RGB -> LMS
	const glm::mat3x3 invMA = kColorTransform_XYZtoRGB_Rec709 * kColorTransform_LMStoXYZ_CAT02; // LMS -> RGB

	// Compute destination reference white in LMS space.
	const glm::vec3 wd = kColorTransform_XYZtoLMS_CAT02 * colorTemperatureToXYZ(6500.f).toGLM();

	// Compute source reference white in LMS space.
	const glm::vec3 ws = kColorTransform_XYZtoLMS_CAT02 * colorTemperatureToXYZ(T).toGLM();

	// Derive final 3x3 transform in RGB space.
	glm::vec3 scale = wd / ws;
	glm::mat3x3 D = glm::diagonal3x3(scale);

	return invMA * D * MA;
}	

__device__ static glm::mat3x3 currentWhiteTransform;
__device__ static float curTemp = 6000.f;
__device__ static bool initializedWhiteBal = false;

__device__ static void initWhiteBalance() {
	currentWhiteTransform = calculateWhiteBalanceTransformRGB_Rec709(6000.f);

	kColorTransform_LMStoXYZ_CAT02 = {
		1.096123820835514, 0.454369041975359, -0.009627608738429,
		-0.278869000218287, 0.473533154307412, -0.005698031216113,
		0.182745179382773, 0.072097803717229, 1.015325639954543
	};

	kColorTransform_XYZtoRGB_Rec709 =
	{
		3.2409699419045213, -0.9692436362808798, 0.0556300796969936,
		-1.5373831775700935, 1.8759675015077206, -0.2039769588889765,
		-0.4986107602930033, 0.0415550574071756, 1.0569715142428784
	};

	kColorTransform_RGBtoXYZ_Rec709 =
	{
		0.4123907992659595, 0.2126390058715104, 0.0193308187155918,
		0.3575843393838780, 0.7151686787677559, 0.1191947797946259,
		0.1804807884018343, 0.0721923153607337, 0.9505321522496608
	};

	kColorTransform_XYZtoLMS_CAT02 =
	{
		0.7328, -0.7036, 0.0030,
		0.4296, 1.6975, 0.0136,
		-0.1624, 0.0061, 0.9834
	};
}

typedef DWORD D3DCOLOR;
#define CUDA_COLOR_TO_DX(r, g, b) ((((0xff) & 0xff) << 24) | (((r) & 0xff) << 16) | (((g) & 0xff) << 8) | ((b) & 0xff));

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

	__global__ void tonemap(float* framebuffer, Camera mainCam, float* postFB, DWORD* dxFB, float* bloomFB, int width, int height, float whiteBalance) {
		// FIRSTLY!! CHECK IF OUR WHITE BALANCE HAS BEEN INVALIDATED
		if (!initializedWhiteBal) {
			initWhiteBalance();
			initializedWhiteBal = true;
		}

		if (whiteBalance != curTemp) {
			// Rebuild the transform
			currentWhiteTransform = calculateWhiteBalanceTransformRGB_Rec709(whiteBalance);
			curTemp = whiteBalance;
		}

		/*
		ACES Approximation by Krzysztof Narkowicz
		https://64.github.io/tonemapping/#aces
		*/
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int pixel_index = j * width * 3 + i * 3;
		int dx_index = j * width + i;

		vec3 frameColor = vec3(framebuffer[pixel_index], framebuffer[pixel_index + 1], framebuffer[pixel_index + 2]) * mainCam.exposure;
		vec3 bloomColor = vec3(bloomFB[pixel_index], bloomFB[pixel_index + 1], bloomFB[pixel_index + 2]);

		//frameColor = (frameColor + bloomColor) / 2;
		// TODO: ADD WHITE BALANCE CONFIG

		// apply white balance
		glm::vec3 newColor = frameColor.toGLM() * currentWhiteTransform;
		frameColor = vec3(newColor.x, newColor.y, newColor.z);

		frameColor *= 0.5f;
		float a = 2.51f;
		float b = 0.03f;
		float c = 1.43f;
		float d = 0.59f;
		float e = 0.14f;
		vec3 tonemapped = (frameColor * (a * frameColor + vec3(b, b, b))) / (frameColor * (c * frameColor + vec3(d, d, d)) + vec3(e, e, e));

		tonemapped.clamp();

		tonemapped = LinearTosRGB(tonemapped);
		
		int r = static_cast<int>(tonemapped.r() * 255.99);
		int g = static_cast<int>(tonemapped.g() * 255.99);
		int Xb = static_cast<int>(tonemapped.b() * 255.99);

		DWORD newColorDX = CUDA_COLOR_TO_DX(r, g, Xb);

		dxFB[dx_index] = newColorDX;
		
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

__host__ void ApplyPostprocess(int width, int height, dim3 blocks, dim3 threads, bool denoiseImage, float whiteBalance) {
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

	tonemap << <blocks, threads >> > (DXHook::fb, DXHook::mainCam, DXHook::postFB, DXHook::dxFB, DXHook::bloomFB, width, height, whiteBalance);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (denoiseImage) {
		denoise << <blocks, threads >> > (DXHook::gbufferData, DXHook::postFB, DXHook::dxFB, DXHook::postFB, width, height);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}
