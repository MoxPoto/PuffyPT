#ifndef TEXTURE_H
#define TEXTURE_H

#include "cuda_runtime.h"
#include "../vec3.cuh"

#include <map>
#include <string>

namespace Tracer {
	typedef float Pixel;

	extern std::map<std::string, Pixel*> deviceTextures;

	class Texture {
	public:
		int resX = 0;
		int resY = 0;
		Pixel* imageData = NULL;
		vec3 fallbackColor;
		bool initialized = false;

		__host__ __device__ Texture();
		__host__ __device__ void SetFallbackColor(vec3 newColor);
		__host__ __device__ void Initialize(int newResX = 0, int newResY = 0, Pixel* newImageData = NULL); // remember, malloc doesn't invoke the constructor
		__host__ __device__ vec3 GetPixel(float u, float v);
	};

	// returns the dev ptr to the texture 
	extern __host__ bool IsTextureCached(const std::string& textureName);
	extern __host__ Pixel* RetrieveCachedTexture(const std::string& textureName);
	extern __host__ Pixel* CreateTextureOnDevice(Pixel* hostPtr, const std::string& textureName, size_t textureSize);
}
#endif