#include "hdri.cuh"
#include "hdriUtility.cuh"

#include <iostream>
#include <vector>
#include <filesystem>
#include "../vendor/stb_image.h"
#include "../util/macros.h"
#include "../dxhook/mainHook.h"

#define HDRI_RESX 1024
#define HDRI_RESY 512

__global__ void createHDRIGPU(Tracer::HDRI* targetHDRI, float* imageData, int resX, int resY) {

    if (imageData == nullptr) {
        NULLPTR_HIT("createHDRIGPU: hit a nullptr on imageData!!");
    }

    // (targetHDRI)->loadData(imageData);
    (targetHDRI)->resX = resX;
    (targetHDRI)->resY = resY;
    (targetHDRI)->brightness = 1.f; // something for me to remember; malloc does not invoke my constructor..
}

namespace Tracer {
	__host__ bool LoadHDRI(const std::string& path) {
        int width = HDRI_RESX;
        int height = HDRI_RESY;
        int comps = 3;
        float* hdriImg = stbi_loadf(path.c_str(), &width, &height, &comps, 3);
        size_t imageSize = 3 * (width * height) * sizeof(float);

        if (hdriImg != NULL) {
            HOST_DEBUG("Loaded HDRI, copying to VRAM..");
            
            checkCudaErrors(cudaMemcpy(DXHook::hdriData, hdriImg, imageSize, cudaMemcpyHostToDevice));
            HOST_DEBUG("Done, instantiating HDRI on gpu now..");
            std::cout << "[host]: image size = " << sizeof(hdriImg) << ", imageSize = " << imageSize << "\n";

            createHDRIGPU << <1, 1 >> > (DXHook::mainHDRI, DXHook::hdriData, width, height);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            HOST_DEBUG("HDRI created on gpu with image data intact, continuing setup..");
        }
        else {
            NULLPTR_HIT("Hit nullptr on hdriImg!!");
            return false;
        }

        stbi_image_free(hdriImg);
	}

}