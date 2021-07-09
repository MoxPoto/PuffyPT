#include <classes/vec3.cuh>
#include <images/texture.cuh>

#include <util/macros.h>
#include <dxhook/mainHook.h>

#include <map>
#include <string>

#include "cuda_runtime.h"

#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

std::map<std::string, Pixel*> deviceTextures;

__device__ Texture::Texture() {
	resX = 0;
	resY = 0;
	imageData = NULL;
	fallbackColor = vec3(0, 0, 0);
}

__device__ void Texture::SetFallbackColor(vec3 newColor) {
	fallbackColor = newColor;
}

__device__ void Texture::Initialize(int newResX, int newResY, Pixel* newImageData) {
	if (newImageData == nullptr)
		return;

	resX = newResX;
	resY = newResY;
	imageData = newImageData;
	initialized = true;
}

__device__ vec3 Texture::GetPixel(float u, float v) {
	if (imageData == nullptr)
		return fallbackColor;

			
	int x = static_cast<int>(fminf(u * resX, resX - 1));
	int y = static_cast<int>(fminf(v * resY, resY - 1));


	int base_index = (3 * (y * resY + x));

	return vec3(imageData[base_index], imageData[base_index + 1], imageData[base_index + 2]);
}

// texture management

__host__ bool IsTextureCached(const std::string& textureName) {
	auto foundValue = deviceTextures.find(textureName);

	return foundValue != deviceTextures.end();
}

__host__ Pixel* RetrieveCachedTexture(const std::string& textureName) {
	if (!IsTextureCached(textureName))
		return nullptr;

	// dev in this case stands for device
	Pixel* devTexture = nullptr;

	try {
		devTexture = deviceTextures.at(textureName);
	}
	catch (std::exception& exception) {
		HOST_DEBUG("Hit a exception; exception reads: %s", exception.what());
		
		return nullptr;
	}

	return devTexture;
}

__host__ Pixel* CreateTextureOnDevice(Pixel* hostPtr, const std::string& textureName, size_t textureSize) {
	if (IsTextureCached(textureName)) {
		return RetrieveCachedTexture(textureName);
	}

	Pixel* devPtr;

	checkCudaErrors(cudaMalloc((void**)&devPtr, textureSize));
	checkCudaErrors(cudaMemcpy(devPtr, hostPtr, textureSize, cudaMemcpyHostToDevice));

	deviceTextures[textureName] = devPtr;

	HOST_DEBUG("Successfully created texture %s on GPU!", textureName.c_str());

	return devPtr;
}
