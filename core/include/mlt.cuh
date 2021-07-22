#ifndef MLT_H
#define MLT_H

#include <dxhook/mainHook.h>
#include "cuda_runtime.h"

extern __device__ vec3 metropolis(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state);

#endif