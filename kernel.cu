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
#include "camera.cuh"

#include "util/macros.h"
#include "brdfs/lambert.cuh"
#include "brdfs/specular.cuh"
#include "brdfs/refraction.cuh"
#include "images/hdriUtility.cuh"

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

__device__ float deg2rad(const float& degree) {
    return degree * M_PI / 180.f;
}

__device__ Tracer::vec3 genSkyColor(Tracer::HDRI* mainHDRI, Tracer::SkyInfo skyInfo, float* imgData, const Tracer::vec3& dir) {
    using namespace Tracer;

    /*
    float t = 0.5f * (dir.z() + 1.0f);
    vec3 skyColor = (1.0f - t) * skyInfo.azimuth + t * skyInfo.zenith;
    */
    
    vec3 skyColor = mainHDRI->getPixelFromRay(dir, imgData);
    
    return skyColor;
}

__device__ Tracer::Object* traceScene(int count, Tracer::Object** world, const Tracer::Ray& ray, Tracer::HitResult& output, bool aabbOverride = false) {
    using namespace Tracer;

    Object* hitObject = NULL;

    output.t = FLT_MAX;

    for (int i = 0; i < count; i++) {
        Tracer::Object* target = *(world + i);

        if (i == ray.ignoreID) continue;

        if (target->anyHit(ray)) {
            // ok, then we trace the precise mesh

            if (target->tryHit(ray, output)) {
                hitObject = target;
            }
        }
    }

    // Fix our shading normal and compute HitPos
    if (hitObject != NULL) {
        output.HitPos = ray.origin + (ray.direction * output.t);

        bool inverted = dot(ray.direction, output.HitNormal) > 0.f;
        output.backface = inverted;

        if (inverted) {
            output.HitNormal = -output.HitNormal;
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
            float lightPower = 2760.f + ((light->emission - EMISSIVE_MINIMUM) * 2.f); // The more intense emission is, more range is added
            float lightBrightness = 1.f;

            vec3 newOrigin = rec.HitPos + (rec.HitNormal * 0.001f);
            vec3 testDirection = unit_vector((light->position - newOrigin));

            Ray testRay(newOrigin, testDirection);
            HitResult testResult;

            Tracer::Object* hitObject = traceScene(count, world, testRay, testResult);


            // A path from the sampled position and the light has been found
            if (hitObject != NULL && hitObject->objectID == light->objectID && testResult.t <= lightPower) {
                // float normalizedRange = (distance / lightPower);

                float falloff = lightPower / ((0.01 * 0.01) + powf(testResult.t, 2.f));

                vec3 lightContribution = (light->getColor(testResult) * falloff) * lightBrightness;

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

__device__ Tracer::vec3 depthColor(DXHook::RenderOptions* options, const Tracer::Ray& ray, curandState* local_rand_state) {
    using namespace Tracer;

    Ray cur_ray = ray;
    vec3 currentLight(1, 1, 1);
    float pdf = 1.f / (2.f * M_PI);


    for (int i = 0; i < options->max_depth; i++) {
        HitResult rec;
        Tracer::Object* target = traceScene(options->count, options->world, cur_ray, rec);

        if (target != NULL) {
            // set our current ray to the new formulated one (this being perfect diffuse)
            // and attenuate our color by the albedo we hit, but we also should multiply our albedo by the objects emission

            if (target->emission > EMISSIVE_MINIMUM) {
                // just return the light
                return currentLight * (target->getColor(rec) * target->emission);
            }

            Ray new_ray(vec3(0, 0, 0), vec3(0, 0, 0));
            vec3 attenuation = currentLight;
            float pdf = 1.f;

            switch (target->matType) {
                case (BRDF::Lambertian):
                    LambertBRDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, attenuation, new_ray, target);
                    break;
                case (BRDF::Specular):
                    SpecularBRDF::SampleWorld(rec, local_rand_state, options->curtime, cur_ray, attenuation, new_ray, target);
                    break;
                case (BRDF::Refraction):
                    RefractBRDF::SampleWorld(rec, local_rand_state, pdf, options->curtime, cur_ray, attenuation, new_ray, target);
                    break;
                default:
                    break;
            }

            currentLight *= attenuation / pdf;
            
            // russian roulette to terminate paths that barely contain any visible contribution
            // from: https://computergraphics.stackexchange.com/a/5808

            /*
            float prob = max(currentLight.x(), max(currentLight.y(), currentLight.z()));

            if (curand_uniform(local_rand_state) > prob) {
                return currentLight;
            }

            // ok, now we add the energy lost from russian rouletting:
            currentLight *= 1 / prob;
            */

            cur_ray = new_ray;
            
        }
        else {
            // didnt hit, finish our depth trace by attenuating our final hit color by the sky color

            if (options->doSky) {
                vec3 skyColor = genSkyColor(options->hdri, options->skyInfo, options->hdriData, cur_ray.direction);

                return (currentLight * (skyColor));
            }
            else {
                return (currentLight * vec3(0.3, 0.3, 0.3));
            }
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ Tracer::vec3 pathtrace(DXHook::RenderOptions* options, const Tracer::Ray& ray, curandState* local_rand_state) {
    using namespace Tracer;
    vec3 indirectLighting(0, 0, 0);
    vec3 directLighting(0, 0, 0); 

    HitResult result;
    Tracer::Object* hitObject = traceScene(options->count, options->world, ray, result);

    if (hitObject != NULL) {
        directLighting = calcDirect(options->count, options->world, hitObject, ray, result);
    }

    for (int i = 0; i < options->samples; i++) {
        indirectLighting += depthColor(options, ray, local_rand_state);
    }

    indirectLighting /= (float)options->samples;


    if (options->curPass == 0) { // Direct only
        return directLighting;  
    }
    else if (options->curPass == 1) { // Indirect only
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

    rotationMat = glm::rotate(rotationMat, glm::radians(-options.cameraDir.x()), glm::vec3(0, 1, 0));
    rotationMat = glm::rotate(rotationMat, glm::radians(options.cameraDir.y()), glm::vec3(0, 0, 1));
    //    rotationMat = glm::rotate(rotationMat, glm::radians(options.roll), glm::vec3(1, 0, 0));


        /*
        vec3 xaxis = cross(vec3(0, 0, 1.f), options.cameraDir);
        xaxis.make_unit_vector();

        vec3 yaxis = cross(options.cameraDir, xaxis);
        yaxis.make_unit_vector();

        rotationMat[0][0] = xaxis.x();
        rotationMat[0][1] = yaxis.x();
        rotationMat[0][2] = options.cameraDir.x();

        rotationMat[1][0] = xaxis.y();
        rotationMat[1][1] = yaxis.y();
        rotationMat[1][2] = options.cameraDir.y();

        rotationMat[2][0] = xaxis.z();
        rotationMat[2][1] = yaxis.z();
        rotationMat[2][2] = options.cameraDir.z();
        */

    glm::vec4 preVec = rotationMat * glm::vec4(dir.x(), dir.y(), dir.z(), 0);

    dir = vec3(preVec.x, preVec.y, preVec.z);

    vec3 origin(options.x, options.y, options.z);

    Ray ourRay(origin, dir);

    HitResult result;
    Tracer::Object* hitObject = traceScene(options.count, options.world, ourRay, result);

    int samples = options.samples;
    int max_depth = options.max_depth;

    // while we're here, let's update our HDRI's brightness as told to by the Host
    options.hdri->brightness = options.hdriBrightness;
    
    if (hitObject != NULL) {
        Ray newRay = ourRay;
        newRay.origin = newRay.origin + (result.HitNormal * 0.001f);

        vec3 indirect = pathtrace(&options, newRay, &local_rand_state);
        indirect.clamp();

        r = (indirect.r());
        g = (indirect.g());
        b = (indirect.b());
    }
    else {
        if (options.doSky) {
            vec3 skyColor = genSkyColor(options.hdri, options.skyInfo, options.hdriData, dir);

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

    // Accumulation can give way to NaN frames which result in black dots
    // so, check if our new pixel is nan--if it is, then restore old frame

    if (isnan(accumulated.x()) || isnan(accumulated.y()) || isnan(accumulated.z()))
        accumulated = previousFrame;

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



__global__ void freeMem(Tracer::Object** world, Tracer::vec3* origin, int worldCount) {
    for (int i = 0; i < worldCount; i++) {
        delete* (world + i);
    }

    delete origin;
}

GMOD_MODULE_OPEN()
{
    using namespace Tracer;

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
    size_t world_size = 90 * sizeof(Tracer::Object*);
    size_t origin_size = sizeof(Tracer::vec3*);
    size_t gbuffer_size = num_pixels * sizeof(Tracer::Post::GBuffer);
    size_t imageSize = 3 * (HDRI_RESX * HDRI_RESY) * sizeof(float);
    size_t hdriSize = sizeof(Tracer::HDRI*);

    HOST_DEBUG("Calculated sizes..");

    checkCudaErrors(cudaMallocManaged((void**)&DXHook::fb, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::postFB, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::world, world_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::origin, origin_size));

    checkCudaErrors(cudaMalloc((void**)&DXHook::gbufferData, gbuffer_size));
    checkCudaErrors(cudaMalloc((void**)&DXHook::d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::hdriData, imageSize));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::mainHDRI, hdriSize));

    HOST_DEBUG("Allocated all memory");

    /*
    DXHook::initMem << <1, 1 >> > (DXHook::world, DXHook::origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    */

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
    using namespace Tracer;

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
    checkCudaErrors(cudaFree(DXHook::hdriData));

    for (std::pair<std::string, Pixel*> devPtr : deviceTextures) {
        HOST_DEBUG("Cleaning %s", devPtr.first.c_str());

        checkCudaErrors(cudaFree(devPtr.second));
    }

    deviceTextures.clear();

    HOST_DEBUG("Freeing DirectX Resources..");
    DXHook::quadVertexBuffer->Release();
    DXHook::msgFont->Release();
    DXHook::pathtraceObject->Release();
    DXHook::pathtraceOutput->Release();
    HOST_DEBUG("Done!");

    cudaDeviceReset();

    HOST_DEBUG("Cuda context freed, module down!");

    FreeConsole();
        
    return 0;
}
