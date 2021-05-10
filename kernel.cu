#include <GarrysMod/Lua/Interface.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

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

#include "brdfs/lambert.cuh"
#include "brdfs/specular.cuh"

#include "dxhook/mainHook.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define WIDTH 480
#define HEIGHT 270
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

void DXHook::check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";

    }
}

__device__ float deg2rad(const float& degree) {
    return degree * M_PI / 180.f;
}

__device__ Tracer::vec3 genSkyColor(const Tracer::vec3& dir) {
    using namespace Tracer;

    float t = 0.5f * (dir.z() + 1.0f);
    vec3 skyColor = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);

    return skyColor;
}

__device__ Tracer::Object* traceScene(int count, Tracer::Object** world, const Tracer::Ray& ray, Tracer::HitResult& output) {
    using namespace Tracer;

    float t_max = FLT_MAX;
    float minDistance = 0.001f;

    Object* hitObject = NULL;

    for (int i = 0; i < count; i++) {
        Tracer::Object* target;

        if (i == 0) {
            target = *(world);
        }
        else {
            target = *(world + i);
        }

        if (target->tryHit(ray, t_max, output) && output.t > minDistance && output.t < t_max) {
            t_max = output.t;
            hitObject = target;
        }
    }

    return hitObject;
}


__device__ Tracer::vec3 depthColor(int count, const Tracer::Ray& ray, Tracer::Object** world, curandState* local_rand_state, int max_depth) {
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
                    LambertBRDF::SampleWorld(rec, local_rand_state, attenuation, new_ray, target);
                    break;
                case (BRDF::Specular):
                    SpecularBRDF::SampleWorld(rec, cur_ray, attenuation, new_ray, target);
                    break;
                default:
                    break;
            }

            currentLight *= attenuation;

            cur_ray = new_ray;
            
        }
        else {
            // didnt hit, finish our depth trace by attenuating our final hit color by the sky color
            vec3 skyColor = genSkyColor(cur_ray.direction);
            
            return (currentLight * (skyColor * 0.04f)) / pdf;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ Tracer::vec3 pathtrace(int count, Tracer::Object** world, const Tracer::Ray& ray, curandState* local_rand_state, int samples, int max_depth) {
    using namespace Tracer;
    vec3 indirectLighting(0, 0, 0);
    vec3 directLighting(0, 0, 0); // to be done soon

    HitResult result;
    Tracer::Object* hitObject = traceScene(count, world, ray, result);

    for (int i = 0; i < samples; i++) {
        indirectLighting += depthColor(count, ray, world, local_rand_state, max_depth);
    }

    indirectLighting /= (float)samples;


    return indirectLighting;
}

__global__ void DXHook::render(float* frameBuffer, Tracer::Object** world, float x, float y, float z, float pitch, float yaw, float roll, curandState* rand_state, int count, float fov, int max_x, int max_y) {
    using namespace Tracer;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    int random_idx = j * max_x + i;

    curandState local_rand_state = rand_state[random_idx];

    float r = 0.f;
    float g = 0.f;
    float b = 0.f;

    float DISTANCE = 1.f;

    float coeff = DISTANCE * tan((fov / 2) * (M_PI / 180)) * 2;
    vec3 camOrigin = vec3(
        DISTANCE,
        (static_cast<float>(max_x - i) / static_cast<float>(max_x - 1) - 0.5) * coeff,
        (coeff / static_cast<float>(max_x)) * static_cast<float>(max_y - j) - 0.5 * (coeff / static_cast<double>(max_x)) * static_cast<double>(max_y - 1)
    );
    vec3 dir = unit_vector(camOrigin);
    // NOT MY CODE!! https://github.com/100PXSquared/public-starfalls/tree/master/raytracer
  
    glm::mat4 rotationMat(1.f);
    // X is roll..
    // Z is yaw
    // so Y is pitch!! YAY!! SOMETHING SORT OF SENSIBLE!!

    rotationMat = glm::rotate(rotationMat, glm::radians(pitch), glm::vec3(0, 1, 0));
    rotationMat = glm::rotate(rotationMat, glm::radians(yaw), glm::vec3(0, 0, 1));
    rotationMat = glm::rotate(rotationMat, glm::radians(roll), glm::vec3(1, 0, 0));

    glm::vec4 preVec = rotationMat * glm::vec4(dir.x(), dir.y(), dir.z(), 0);
    
    dir = vec3(preVec.x, preVec.y, preVec.z);

    vec3 origin(x, y, z);

    Ray ourRay(origin, dir);

    HitResult result;
    Tracer::Object* hitObject = traceScene(count, world, ourRay, result);

    int samples = 12;
    int max_depth = 3;

    if (hitObject != NULL) {
        Ray newRay = ourRay;
        newRay.origin = newRay.origin + (result.HitNormal * 0.001f);

        vec3 indirect = pathtrace(count, world, newRay, &local_rand_state, samples, max_depth);
        indirect.clamp();

        r = sqrt(indirect.r());
        g = sqrt(indirect.g());
        b = sqrt(indirect.b());
    }
    else {
        vec3 skyColor = genSkyColor(dir);
        
        r = skyColor.r();
        g = skyColor.g();
        b = skyColor.b();
    }

    frameBuffer[pixel_index + 0] = (frameBuffer[pixel_index + 0] + r) / 2.f;
    frameBuffer[pixel_index + 1] = (frameBuffer[pixel_index + 1] + g) / 2.f;
    frameBuffer[pixel_index + 2] = (frameBuffer[pixel_index + 2] + b) / 2.f;
}

__global__ void DXHook::initMem(Tracer::Object** world, Tracer::vec3* origin) {
    using namespace Tracer; 

    origin = (new Tracer::vec3(0, 0, 0));

    *(world) = (new Tracer::Sphere(vec3(10, 0, 0), .2f));
    Tracer::Object* objOne = *(world);
    objOne->color = vec3(1, 1, 1);
    objOne->emission = 1.f;
    objOne->matType = BRDF::Specular;

    *(world + 1) = (new Tracer::Sphere(vec3(10, 0, -3.2), 3.f));
    Tracer::Object* objTwo = *(world + 1);
    objTwo->color = vec3(1.f, 0.5f, 0.5f);
    objTwo->emission = 1.f;

    *(world + 2) = (new Tracer::Sphere(vec3(11, 3, 1), 0.7f));
    Tracer::Object* objThree = *(world + 2);
    objThree->color = vec3(1.f, 1.f, 1.f);
    objThree->emission = 50.f;

}

__global__ void DXHook::registerRands(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984 + pixel_index, pixel_index, 0, &rand_state[pixel_index]);
    
}

__global__ void freeMem(Tracer::Object** world, Tracer::vec3* origin) {
    delete* (world); // to-do actually encapsulate entities in a world
    delete origin;
}

GMOD_MODULE_OPEN()
{
    int num_pixels = WIDTH * HEIGHT;
    size_t fb_size = 3 * num_pixels * sizeof(float);
    size_t world_size = 3 * sizeof(Tracer::Object*);
    size_t origin_size = sizeof(Tracer::vec3*);

    checkCudaErrors(cudaMallocManaged((void**)&DXHook::fb, fb_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::world, world_size));
    checkCudaErrors(cudaMallocManaged((void**)&DXHook::origin, origin_size));

    checkCudaErrors(cudaMalloc((void**)&DXHook::d_rand_state, num_pixels * sizeof(curandState)));

    DXHook::initMem << <1, 1 >> > (DXHook::world, DXHook::origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int warpX = 16;
    int warpY = 16; // technically can be ruled out as tiled rendering

    dim3 blocks(WIDTH / warpX + 1, HEIGHT / warpY + 1);
    dim3 threads(warpX, warpY);

    DXHook::registerRands << < blocks, threads >> > (WIDTH, HEIGHT, DXHook::d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Run all of our starting CUDA code

    DXHook::Initialize(LUA);

    return 0;
}

GMOD_MODULE_CLOSE() 
{
    DXHook::Cleanup(LUA);

    Sleep(2000);

    freeMem << <1, 1 >> > (DXHook::world, DXHook::origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(DXHook::fb));
    checkCudaErrors(cudaFree(DXHook::world));
    checkCudaErrors(cudaFree(DXHook::d_rand_state));
    checkCudaErrors(cudaFree(DXHook::origin));

    DXHook::quadVertexBuffer->Release();
    DXHook::msgFont->Release();
    DXHook::pathtraceObject->Release();
    DXHook::pathtraceOutput->Release();

    cudaDeviceReset();

    return 0;
}
