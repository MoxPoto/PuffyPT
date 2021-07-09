#include <GarrysMod/Lua/Interface.h>
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "math_constants.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>

#include <util/macros.h>
#include <brdfs/lambert.cuh>
#include <brdfs/specular.cuh>
#include <brdfs/refraction.cuh>
#include <images/hdriUtility.cuh>

#include <dxhook/mainHook.h>
#include <postprocess/mainDenoiser.cuh>
#include <cpugpu/objects.cuh>
#include <synchronization/syncMain.cuh>

#include <pathtracer.cuh>

#define STB_IMAGE_IMPLEMENTATION
#include <vendor/stb_image.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define WIDTH 480
#define HEIGHT 270
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )
#define HDRI_LOCATION "C:\\pathtracer\\hdrs\\shanghai_bund_1k.hdr"
#define HDRI_FOLDER "C:\\pathtracer\\hdrs"
#define HDRI_RESX 2048
#define HDRI_RESY 1024

void DXHook::check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << "CUDA_ERROR_STRING: " << cudaGetErrorString(result) << "\n" <<
            cudaGetErrorName(result) << "\n";
    }
}

__global__ void DXHook::render(DXHook::RenderOptions options) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= options.max_x) || (j >= options.max_y)) return;
    int pixel_index = j * options.max_x * 3 + i * 3;
    int random_idx = j * options.max_x + i;

    curandState local_rand_state = options.rand_state[random_idx];

    curand_init(options.frameCount * options.max_x * options.max_y + j * options.max_x + i, 1, 0, &local_rand_state);

    Post::GBuffer* gbuffer = ((options.gbufferPtr + random_idx)); // serves as a gbuffer access index too!!

    vec3 frameColor;

    float DISTANCE = 1.f;

    float coeff = DISTANCE * tan((options.fov / 2) * (M_PI / 180)) * 2;
    vec3 camOrigin = vec3(
        DISTANCE,
        (static_cast<float>(options.max_x - i) / static_cast<float>(options.max_x - 1) - 0.5) * coeff,
        (coeff / static_cast<float>(options.max_x)) * static_cast<float>(options.max_y - j) - 0.5 * (coeff / static_cast<double>(options.max_x)) * static_cast<double>(options.max_y - 1)
    );
    vec3 dir = unit_vector(camOrigin);
    // NOT MY CODE!! https://github.com/100PXSquared/public-starfalls/tree/master/raytracer

    glm::mat4 rotationMat(1.f);

    // X is roll..
    // Z is yaw
    // so Y is pitch!! YAY!! SOMETHING SORT OF SENSIBLE!!

    rotationMat = glm::rotate(rotationMat, glm::radians(-options.cameraDir.x()), glm::vec3(0, 1, 0));
    rotationMat = glm::rotate(rotationMat, glm::radians(options.cameraDir.y()), glm::vec3(0, 0, 1));

    glm::vec4 preVec = rotationMat * glm::vec4(dir.x(), dir.y(), dir.z(), 0);

    dir = vec3(preVec.x, preVec.y, preVec.z);

    vec3 origin(options.x, options.y, options.z);

    Ray ourRay(origin, dir);

    HitResult result;
    Object* hitObject = traceScene(options.count, options.world, ourRay, result);

    int samples = options.samples;
    int max_depth = options.max_depth;

    // while we're here, let's update our HDRI's brightness as told to by the Host
    options.hdri->brightness = options.hdriBrightness;
    
    if (hitObject != NULL) {
        Ray newRay = ourRay;
        newRay.origin = newRay.origin + (result.HitNormal * 0.001f);

        vec3 indirect = pathtrace(&options, newRay, &local_rand_state);

        frameColor = indirect;
    }
    else {
        if (options.doSky) {
            vec3 skyColor = genSkyColor(options.hdri, options.skyInfo, options.hdriData, dir);

            frameColor = skyColor;
        }
    }
    
    if (hitObject != NULL) {
        gbuffer->albedo = hitObject->GetColor(result);
        gbuffer->normal = result.HitNormal;
        gbuffer->objectID = result.objId;
        gbuffer->brdfType = hitObject->matType;
    }
    
    gbuffer->position = result.HitPos;
    gbuffer->depth = result.t;
    gbuffer->diffuse = frameColor;
    gbuffer->isSky = (hitObject == NULL);
    
    vec3 previousFrame = vec3(options.frameBuffer[pixel_index + 0], options.frameBuffer[pixel_index + 1], options.frameBuffer[pixel_index + 2]);
    vec3 accumulated = (frameColor + previousFrame * options.frameCount) / (options.frameCount + 1);

    // Accumulation can give way to NaN frames which result in black dots
    // so, check if our new pixel is nan--if it is, then restore old frame

    if (isnan(accumulated.x()) || isnan(accumulated.y()) || isnan(accumulated.z()))
        accumulated = previousFrame;

    options.frameBuffer[pixel_index + 0] = accumulated.r();
    options.frameBuffer[pixel_index + 1] = accumulated.g();
    options.frameBuffer[pixel_index + 2] = accumulated.b();
}

__global__ void DXHook::registerRands(int max_x, int max_y, curandState* rand_state, Post::GBuffer* gbufferData) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984 + pixel_index, pixel_index, 0, &rand_state[pixel_index]);
    // lets also initialize our GBuffers
    Post::GBuffer myBuffer;

    *(gbufferData + pixel_index) = myBuffer;
}

__global__ void freeMem(Object** world, vec3* origin, int worldCount) {
    for (int i = 0; i < worldCount; i++) {
        delete* (world + i);
    }

    delete origin;
}

GMOD_MODULE_OPEN()
{
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024000 * 10);

    AllocConsole();
    FILE* pFile = nullptr;

    freopen_s(&pFile, "CONOUT$", "w", stdout); // cursed way to redirect stdout to our own console

    DXHook::lastTime = std::chrono::high_resolution_clock::now();

    HOST_DEBUG("Querying device..");
    int ourDeviceID;
    checkCudaErrors(cudaGetDevice(&ourDeviceID));

    HOST_DEBUG("Got device!");
    cudaDeviceProp properties;

    checkCudaErrors(cudaGetDeviceProperties(&properties, ourDeviceID));

    HOST_DEBUG("Got properties..");

    HOST_DEBUG("Using GPU %s\n", properties.name);
    HOST_DEBUG("Is integrated: %d\n", properties.integrated);
    HOST_DEBUG("Max threads per block: %d\n", properties.maxThreadsPerBlock);
    HOST_DEBUG("GPU's MP count: %d\n", properties.multiProcessorCount);
    HOST_DEBUG("Major: %d, Minor: %d", properties.major, properties.minor);

    HOST_DEBUG("Starting memory allocation for GPU");

    int num_pixels = WIDTH * HEIGHT;
    size_t fb_size = 3 * num_pixels * sizeof(float);
    size_t world_size = 260 * sizeof(Object*);
    size_t origin_size = sizeof(vec3*);
    size_t gbuffer_size = num_pixels * sizeof(Post::GBuffer);
    size_t imageSize = 3 * (HDRI_RESX * HDRI_RESY) * sizeof(float);
    size_t hdriSize = sizeof(HDRI*);

    HOST_DEBUG("Calculated sizes..");

    checkCudaErrors(cudaMallocManaged((void**)&DXHook::fb, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::postFB, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::bloomFB, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::blurFB, fb_size));

    checkCudaErrors(cudaMallocManaged((void**)&DXHook::world, world_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::origin, origin_size));

    checkCudaErrors(cudaMalloc((void**)&DXHook::gbufferData, gbuffer_size));
    checkCudaErrors(cudaMalloc((void**)&DXHook::d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::hdriData, imageSize));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::mainHDRI, hdriSize));

    HOST_DEBUG("Allocated all memory");

    HOST_DEBUG("Reading HDRI from disk..");
    
    bool correctLoad = LoadHDRI(HDRI_LOCATION);

    if (!correctLoad) {
        HOST_DEBUG("Loading HDRI failed! Not continuing tracer loading..");
        return 0;
    }

    FindHDRIs(HDRI_FOLDER, DXHook::hdriList, DXHook::hdriListSize);

    for (int i = 0; i < DXHook::hdriListSize; i++) {
        std::string path = DXHook::hdriList.at(i);

        if (path == HDRI_LOCATION) {
            DXHook::curHDRI = i;
            break;
        }
    }

    HOST_DEBUG("Starting random threads..");

    int warpX = 16;
    int warpY = 16; // technically can be ruled out as tiled rendering

    dim3 blocks(WIDTH / warpX + 1, HEIGHT / warpY + 1);
    dim3 threads(warpX, warpY);

    ClearFramebuffer << <blocks, threads >> > (DXHook::fb, WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ClearFramebuffer << <blocks, threads >> > (DXHook::postFB, WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ClearFramebuffer << <blocks, threads >> > (DXHook::bloomFB , WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    DXHook::registerRands << < blocks, threads >> > (WIDTH, HEIGHT, DXHook::d_rand_state, DXHook::gbufferData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    HOST_DEBUG("Finished!");
    // Run all of our starting CUDA code

    HOST_DEBUG("Starting DXHook..");
    DXHook::Initialize(LUA);
    HOST_DEBUG("Finished!");

    HOST_DEBUG("Starting Synchronization Service..");
    Sync::Initialize(LUA);
    HOST_DEBUG("Finished!");

    return 0;
}

GMOD_MODULE_CLOSE() 
{
    HOST_DEBUG("Closing module!");
    HOST_DEBUG("Closing DXHook..");
    DXHook::Cleanup(LUA);
    HOST_DEBUG("Finished!");

    Sync::Deinitialize(LUA);
    HOST_DEBUG("Closed Sync..");

    HOST_DEBUG("Freeing GPU memory, closing CUDA context..");
   
    Sleep(2000);

    freeMem << <1, 1 >> > (DXHook::world, DXHook::origin, DXHook::world_count);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(DXHook::fb));
    checkCudaErrors(cudaFree(DXHook::world));
    checkCudaErrors(cudaFree(DXHook::d_rand_state));
    checkCudaErrors(cudaFree(DXHook::origin));
    checkCudaErrors(cudaFree(DXHook::gbufferData));
    checkCudaErrors(cudaFree(DXHook::blurFB));
    checkCudaErrors(cudaFree(DXHook::bloomFB));
    checkCudaErrors(cudaFree(DXHook::postFB));

    for (std::pair<std::string, Pixel*> devPtr : deviceTextures) {
        HOST_DEBUG("Cleaning %s", devPtr.first.c_str());

        checkCudaErrors(cudaFree(devPtr.second));
    }

    deviceTextures.clear();

    HOST_DEBUG("Freeing DirectX Resources..");
    DXHook::msgFont->Release();
    DXHook::pathtraceObject->Release();
    DXHook::pathtraceOutput->Release();
    HOST_DEBUG("Done!");

    cudaDeviceReset();

    HOST_DEBUG("Cuda context freed, module down!");
    HOST_DEBUG("You can close this window now!");
    FreeConsole();
        
    return 0;
}
