#include <pathtracer/pathtracer.cuh>
#include <renderer/render.cuh>

#include <cuda_runtime.h>
#include <imgui.h>

#include <iostream>
#include <d3dx9.h>

__host__ void Pathtracer::Update() {
    updateMutex.lock();
    int tileX = 6;
    int tileY = 6;

    dim3 blocks(width / tileX + 1, height / tileY + 1);
    dim3 threads(tileX, tileY);

    renderKernel << <blocks, threads >> > (framebuffer, dxFramebuffer, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Then, we update the texture
    D3DLOCKED_RECT textureData;
    // TODO: Add error checking
    renderTexture->LockRect(0, &textureData, NULL, 0);
    memcpy(reinterpret_cast<DWORD*>(textureData.pBits), (void*)dxFramebuffer, width * height * sizeof(DWORD));
    renderTexture->UnlockRect(0);

    updateMutex.unlock();
}

__host__ void Pathtracer::ImGuiUpdate() {
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
}

__host__ void Pathtracer::ErrorCheck(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << "CUDA_ERROR_STRING: " << cudaGetErrorString(result) << "\n" <<
            cudaGetErrorName(result) << "\n";
    }
}

__host__ void Pathtracer::Allocate(void* gpuMemory, size_t bufferSize, bool managed) {
    if (managed) {
        checkCudaErrors(cudaMallocManaged((void**)&gpuMemory, bufferSize));
    }
    else {
        checkCudaErrors(cudaMalloc((void**)&gpuMemory, bufferSize));
    }

    buffersToRelease.push_back(gpuMemory);
}

__host__ Pathtracer::Pathtracer(int _width, int _height, LPDIRECT3DDEVICE9 device) {
    width = _width;
    height = _height;

    size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    Allocate(framebuffer, num_pixels * 3 * sizeof(float), true);
    Allocate(dxFramebuffer, num_pixels * sizeof(DWORD), true);

    renderTexture = NULL;
    HRESULT code = D3DXCreateTexture(device, width, height, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &renderTexture);

    if (!renderTexture) {
        printf("Couldn't create the render texture!!\nCode: %u\n", static_cast<unsigned int>(code));
        return;
    }

    valid = true;
}

__host__ Pathtracer::~Pathtracer() {
    updateMutex.lock();
    // Release all the buffers we allocated
    for (void* buffer : buffersToRelease) {
        cudaFree(buffer);
    }
    
    valid = false; // Invalidate the class
    renderTexture->Release();

    updateMutex.unlock();
}