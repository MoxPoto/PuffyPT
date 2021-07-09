#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "cuda_runtime.h"

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/object.cuh>

#include <dxhook/mainHook.h>

// TODO: this shouldnt be in the pathtracer file, but it'll do for now since pretty much nothing else requires it
// however - in something like refraction medium swapping, the BRDF will need to be able to trace, so that requires a new
// file for this function
extern __device__ Object* traceScene(int count, Object** world, const Ray& ray, HitResult& output, bool aabbOverride = false);
extern __device__ vec3 genSkyColor(HDRI* mainHDRI, SkyInfo skyInfo, float* imgData, const vec3& dir);

extern __device__ vec3 pathtrace(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state);

#endif