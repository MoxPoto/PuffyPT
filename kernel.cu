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

#include "ray.cuh"
#include "mesh.cuh"
#include "sphere.cuh"
#include "vec3.cuh"
#include "object.cuh"
#include "triangle.cuh"
#include "hitresult.cuh"

#include "util/macros.h"
#include "brdfs/lambert.cuh"
#include "brdfs/specular.cuh"

#include "dxhook/mainHook.h"
#include "postprocess/mainDenoiser.cuh"
#include "cpugpu/objects.cuh"
#include "synchronization/syncMain.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb_image.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define WIDTH 480
#define HEIGHT 270
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )
#define DEBUGHOST(str) printf("[host]: %s\n", str);
#define HDRI_LOCATION "C:\\pathtracer\\hdrs\\noon_grass_1k.hdr"
#define HDRI_RESX 1024
#define HDRI_RESY 512

void DXHook::check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << "CUDA_ERROR_STRING: " << cudaGetErrorString(result) << "\n" <<
            cudaGetErrorName(result) << "\n";

    }
}

__device__ float deg2rad(const float& degree) {
    return degree * M_PI / 180.f;
}

__device__ Tracer::vec3 genSkyColor(Tracer::HDRI* mainHDRI, float* imgData, const Tracer::vec3& dir) {
    using namespace Tracer;

    
    float t = 0.5f * (dir.z() + 1.0f);
    vec3 skyColor = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    
    /*
    vec3 skyColor = mainHDRI->getPixelFromRay(dir, imgData);
    */
    return skyColor;
}

__device__ Tracer::Object* traceScene(int count, Tracer::Object** world, const Tracer::Ray& ray, Tracer::HitResult& output) {
    using namespace Tracer;

    float t_max = FLT_MAX;
    float approxtMax = FLT_MAX;
    float minDistance = FLT_MIN;

    Object* hitObject = NULL;
   

    for (int i = 0; i < count; i++) {
        Tracer::Object* target;

        if (i == 0) {
            target = *(world);
        }
        else {
            target = *(world + i);
        }

        if (target->objectID == ray.ignoreID) {
            continue;
        }

        float placeholdertMax = approxtMax;

        if (target->anyHit(ray, placeholdertMax)) { 
            // ok, then we trace the precise mesh

            if (target->tryHit(ray, t_max, output) && output.t > minDistance && output.t < t_max) {
                t_max = output.t;
                hitObject = target;
                output.objId = i;
            }
        }
    }

    return hitObject;
}

// to-do: clean this damn shit up.. like come on.. a simple shading.cu file would do, like dam..
__device__ const int EMISSIVE_MINIMUM = 15; // Minimum emission to be considered a light
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

__device__ Tracer::vec3 calcDirect(int count, Tracer::Object** world, Tracer::Object* firstHit, const Tracer::Ray& ray, const Tracer::HitResult& rec) {
    using namespace Tracer;

    vec3 lightObtained(0, 0, 0);
    int lightHits = 0;

    for (int i = 0; i < count; i++) {
        Tracer::Object* light = *(world + i);

        if (light->emission >= EMISSIVE_MINIMUM) {
            float lightPower = 15.f;// +((light->emission - EMISSIVE_MINIMUM) * 2.f); // The more intense emission is, more range is added
            
            vec3 newOrigin = rec.HitPos + (rec.HitNormal * 0.001f);
            vec3 testDirection = unit_vector((light->position - newOrigin));

            Ray testRay(newOrigin, testDirection);
            HitResult testResult;

            Tracer::Object* hitObject = traceScene(count, world, testRay, testResult);

            float distance = testResult.t;

            // A path from the sampled position and the light has been found
            if (hitObject != NULL && hitObject->objectID == light->objectID && testResult.t <= distance) {
                float clampedRange = distance;
                float normalizedRange = (clampedRange / lightPower);

                float invRange = (1.0f - normalizedRange);

                vec3 lightContribution = light->color * invRange;

                lightHits++;
                lightObtained += lightContribution;
            }
        }
    }

    if (lightHits == 0) {
        return lightObtained;
    }
    else {
        lightObtained /= static_cast<float>(lightHits);
        return lightObtained;
    }

}

__device__ Tracer::vec3 depthColor(int count, Tracer::HDRI* mainHDRI, float* imgData, bool doSky, float extraRand, const Tracer::Ray& ray, Tracer::Object** world, curandState* local_rand_state, int max_depth) {
    using namespace Tracer;

    Ray cur_ray = ray;
    vec3 currentLight(1, 1, 1);
    float pdf = 1.f / (2.f * M_PI);


    for (int i = 0; i < max_depth; i++) {
        HitResult rec;
        Tracer::Object* target = traceScene(count, world, cur_ray, rec);

        if (target != NULL) {
            // set our current ray to the new formulated one (this being perfect diffuse)
            // and attenuate our color by the albedo we hit, but we also should multiply our albedo by the objects emission
            Ray new_ray(vec3(0, 0, 0), vec3(0, 0, 0));
            vec3 attenuation(0, 0, 0);

            switch (target->matType) {
                case (BRDF::Lambertian):
                    LambertBRDF::SampleWorld(rec, local_rand_state, extraRand, attenuation, new_ray, target);
                    break;
                case (BRDF::Specular):
                    SpecularBRDF::SampleWorld(rec, local_rand_state, extraRand, cur_ray, attenuation, new_ray, target);
                    break;
                default:
                    break;
            }

            currentLight *= attenuation;

            cur_ray = new_ray;
            
        }
        else {
            // didnt hit, finish our depth trace by attenuating our final hit color by the sky color

            if (doSky) {
                vec3 skyColor = genSkyColor(mainHDRI, imgData, cur_ray.direction);

                return (currentLight * (skyColor * 0.20f));
            }
            else {
                return (currentLight * vec3(0.3, 0.3, 0.3));
            }
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ Tracer::vec3 pathtrace(int count, int currentPass, Tracer::HDRI* mainHDRI, float* imgData, bool doSky, float extraRand, Tracer::Object** world, const Tracer::Ray& ray, curandState* local_rand_state, int samples, int max_depth) {
    using namespace Tracer;
    vec3 indirectLighting(0, 0, 0);
    vec3 directLighting(0, 0, 0); 

    HitResult result;
    Tracer::Object* hitObject = traceScene(count, world, ray, result);

    if (hitObject != NULL) {
        directLighting = calcDirect(count, world, hitObject, ray, result);
    }

    for (int i = 0; i < samples; i++) {
        indirectLighting += depthColor(count, mainHDRI, imgData, doSky, extraRand, ray, world, local_rand_state, max_depth);
    }

    indirectLighting /= (float)samples;


    if (currentPass == 0) { // Direct only
        return directLighting;  
    }
    else if (currentPass == 1) { // Indirect only
        return indirectLighting;
    }
    else {
        return (directLighting / CUDART_PI + 2.0 * indirectLighting);
    }
}

__global__ void DXHook::render(DXHook::RenderOptions options) {
    using namespace Tracer;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= options.max_x) || (j >= options.max_y)) return;
    int pixel_index = j * options.max_x * 3 + i * 3;
    int random_idx = j * options.max_x + i;

    curandState local_rand_state = options.rand_state[random_idx];
    
    curand_init(options.frameCount * options.max_x * options.max_y + j * options.max_x + i, 1, 0, &local_rand_state);

    //Denoising::GBuffer* gbuffer = ((options.gbufferPtr + random_idx)); // serves as a gbuffer access index too!!

    float r = 0.f;
    float g = 0.f;
    float b = 0.f;

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

    rotationMat = glm::rotate(rotationMat, glm::radians(-options.pitch), glm::vec3(0, 1, 0));
    rotationMat = glm::rotate(rotationMat, glm::radians(options.yaw), glm::vec3(0, 0, 1));
//    rotationMat = glm::rotate(rotationMat, glm::radians(options.roll), glm::vec3(1, 0, 0));

    glm::vec4 preVec = rotationMat * glm::vec4(dir.x(), dir.y(), dir.z(), 0);
    
    dir = vec3(preVec.x, preVec.y, preVec.z);

    vec3 origin(options.x, options.y, options.z);

    Ray ourRay(origin, dir);

    HitResult result;
    Tracer::Object* hitObject = traceScene(options.count, options.world, ourRay, result);

    int samples = options.samples;
    int max_depth = options.max_depth;

    if (hitObject != NULL) {
        Ray newRay = ourRay;
        newRay.origin = newRay.origin + (result.HitNormal * 0.001f);

        vec3 indirect = pathtrace(options.count, options.curPass, options.hdri, options.hdriData, options.doSky, options.curtime, options.world, newRay, &local_rand_state, samples, max_depth);
        indirect.clamp();

        r = (indirect.r());
        g = (indirect.g());
        b = (indirect.b());
    }
    else {
        if (options.doSky) {
            vec3 skyColor = genSkyColor(options.hdri, options.hdriData, dir);

            r = skyColor.r();
            g = skyColor.g();
            b = skyColor.b();
        }
    }

    if (hitObject != NULL) {
        /*
        gbuffer->albedo = hitObject->color;
        gbuffer->normal = result.HitNormal;
        gbuffer->objectID = result.objId;
        gbuffer->brdfType = hitObject->matType;
        */
    }

    /*
    gbuffer->position = result.HitPos;
    gbuffer->depth = result.t;
    gbuffer->diffuse = vec3(r, g, b);
    gbuffer->isSky = (hitObject == NULL);
    */

    vec3 previousFrame = vec3(options.frameBuffer[pixel_index + 0], options.frameBuffer[pixel_index + 1], options.frameBuffer[pixel_index + 2]);
    vec3 curFrame = vec3(r, g, b);

    vec3 accumulated = (curFrame + previousFrame * options.frameCount) / (options.frameCount + 1);

    options.frameBuffer[pixel_index + 0] = accumulated.r();
    options.frameBuffer[pixel_index + 1] = accumulated.g();
    options.frameBuffer[pixel_index + 2] = accumulated.b();
}

__global__ void DXHook::initMem(Tracer::Object** world, Tracer::vec3* origin) {
    using namespace Tracer; 

    origin = (new Tracer::vec3(0, 0, 0));

    *(world) = (new Tracer::Sphere(vec3(10, 0, 0), .2f));
    Tracer::Object* objOne = *(world);
    objOne->color = vec3(1, 1, 1);
    objOne->emission = 1.f;
    objOne->lighting.ior = 1.2f;

    *(world + 1) = (new Tracer::Sphere(vec3(10, 0, -3.2), 3.f));
    Tracer::Object* objTwo = *(world + 1);
    objTwo->color = vec3(1.f, 0.5f, 0.5f);
    objTwo->emission = 1.f;

    *(world + 2) = (new Tracer::Sphere(vec3(11, 3, 1), 0.7f));
    Tracer::Object* objThree = *(world + 2);
    objThree->color = vec3(1.f, 1.f, 1.f);
    objThree->emission = 50.f;

}

__global__ void DXHook::registerRands(int max_x, int max_y, curandState* rand_state, Tracer::Post::GBuffer* gbufferData) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984 + pixel_index, pixel_index, 0, &rand_state[pixel_index]);
    // lets also initialize our GBuffers
    Tracer::Post::GBuffer myBuffer;

    *(gbufferData + pixel_index) = myBuffer;
}

__global__ void createHDRIGPU(Tracer::HDRI* targetHDRI, float* imageData) {
    targetHDRI = new Tracer::HDRI();

    if (imageData == nullptr) {
        NULLPTR_HIT("createHDRIGPU: hit a nullptr on imageData!!");
    }

    // (targetHDRI)->loadData(imageData);
    (targetHDRI)->resX = HDRI_RESX;
    (targetHDRI)->resY = HDRI_RESY;
}

__global__ void initializeHDRI(float* hdriData, size_t imageSize) {
    (hdriData) = new float[imageSize];
}

__global__ void freeMem(Tracer::Object** world, Tracer::vec3* origin, int worldCount) {
    for (int i = 0; i < worldCount; i++) {
        delete* (world + i);
    }

    delete origin;
}

GMOD_MODULE_OPEN()
{
    using namespace Tracer;

    AllocConsole();
    FILE* pFile = nullptr;

    freopen_s(&pFile, "CONOUT$", "w", stdout); // cursed way to redirect stdout to our own console

    DXHook::lastTime = std::chrono::high_resolution_clock::now();

    DEBUGHOST("Querying device..");
    int ourDeviceID;
    checkCudaErrors(cudaGetDevice(&ourDeviceID));

    DEBUGHOST("Got device!");
    cudaDeviceProp properties;

    checkCudaErrors(cudaGetDeviceProperties(&properties, ourDeviceID));

    DEBUGHOST("Got properties..");

    printf("[host]: Using GPU %s\n", properties.name);
    printf("[host]: Is integrated: %d\n", properties.integrated);
    printf("[host]: Max threads per block: %d\n", properties.maxThreadsPerBlock);
    printf("[host]: GPU's MP count: %d\n", properties.multiProcessorCount);
    printf("[host]: Major: %d, Minor: %d", properties.major, properties.minor);

    DEBUGHOST("Starting memory allocation for GPU");

    int num_pixels = WIDTH * HEIGHT;
    size_t fb_size = 3 * num_pixels * sizeof(float);
    size_t world_size = 90 * sizeof(Tracer::Object*);
    size_t origin_size = sizeof(Tracer::vec3*);
    size_t gbuffer_size = num_pixels * sizeof(Tracer::Post::GBuffer);
    size_t imageSize = 3 * (HDRI_RESX * HDRI_RESY) * sizeof(float);
    size_t hdriSize = sizeof(Tracer::HDRI*);

    DEBUGHOST("Calculated sizes..");

    checkCudaErrors(cudaMallocManaged((void**)&DXHook::fb, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::postFB, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::world, world_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::origin, origin_size));

    checkCudaErrors(cudaMalloc((void**)&DXHook::gbufferData, gbuffer_size));
    checkCudaErrors(cudaMalloc((void**)&DXHook::d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::hdriData, imageSize));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::mainHDRI, hdriSize));

    DEBUGHOST("Allocated all memory");

    /*
    DXHook::initMem << <1, 1 >> > (DXHook::world, DXHook::origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    */

    DEBUGHOST("Reading HDRI from disk..");
    
    int width = HDRI_RESX;
    int height = HDRI_RESY;
    int comps = 3;
    float* hdriImg = stbi_loadf(HDRI_LOCATION, &width, &height, &comps, 0);


    if (hdriImg != NULL) {
        DEBUGHOST("Loaded HDRI, copying to VRAM..");
        DEBUGHOST("Sample R, G, B:");
        int startIdx = (3 * (1 * HDRI_RESX + 1));
        HOST_DEBUG("R: %.2f, G: %.2f, B: %.2f\n", hdriImg[startIdx], hdriImg[startIdx + 1], hdriImg[startIdx + 2]);

        checkCudaErrors(cudaMemcpy(DXHook::hdriData, hdriImg, imageSize, cudaMemcpyHostToDevice));
        DEBUGHOST("Done, instantiating HDRI on gpu now..");
        std::cout << "[host]: image size = " << sizeof(hdriImg) << ", imageSize = " << imageSize << "\n";
        
        createHDRIGPU << <1, 1 >> > (DXHook::mainHDRI, DXHook::hdriData);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        DEBUGHOST("HDRI created on gpu with image data intact, continuing setup..");
    }
    else {
        NULLPTR_HIT("Hit nullptr on hdriImg!!");
        return 0;
    }

    DEBUGHOST("Starting random threads..");

    int warpX = 16;
    int warpY = 16; // technically can be ruled out as tiled rendering

    dim3 blocks(WIDTH / warpX + 1, HEIGHT / warpY + 1);
    dim3 threads(warpX, warpY);

    DXHook::registerRands << < blocks, threads >> > (WIDTH, HEIGHT, DXHook::d_rand_state, DXHook::gbufferData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    DEBUGHOST("Finished!");
    // Run all of our starting CUDA code

    DEBUGHOST("Starting DXHook..");
    DXHook::Initialize(LUA);
    DEBUGHOST("Finished!");

    DEBUGHOST("Starting Synchronization Service..");
    Sync::Initialize(LUA);
    DEBUGHOST("Finished!");

    DEBUGHOST("Clearing HDRI on CPU since it's on the GPU..");
    stbi_image_free(hdriImg);
    DEBUGHOST("Done!");
    return 0;
}

GMOD_MODULE_CLOSE() 
{
    using namespace Tracer;

    DEBUGHOST("Closing module!");
    DEBUGHOST("Closing DXHook..");
    DXHook::Cleanup(LUA);
    DEBUGHOST("Finished!");

    Sync::Deinitialize(LUA);
    DEBUGHOST("Closed Sync..");

    DEBUGHOST("Freeing GPU memory, closing CUDA context..");
   
    Sleep(2000);

    freeMem << <1, 1 >> > (DXHook::world, DXHook::origin, DXHook::world_count);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(DXHook::fb));
    checkCudaErrors(cudaFree(DXHook::world));
    checkCudaErrors(cudaFree(DXHook::d_rand_state));
    checkCudaErrors(cudaFree(DXHook::origin));
    checkCudaErrors(cudaFree(DXHook::gbufferData));
    checkCudaErrors(cudaFree(DXHook::hdriData));
    DEBUGHOST("Freeing DirectX Resources..");
    DXHook::quadVertexBuffer->Release();
    DXHook::msgFont->Release();
    DXHook::pathtraceObject->Release();
    DXHook::pathtraceOutput->Release();
    DEBUGHOST("Done!");

    cudaDeviceReset();

    DEBUGHOST("Cuda context freed, module down!");

    FreeConsole();
        
    return 0;
}
