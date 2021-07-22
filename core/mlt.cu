#include <dxhook/mainHook.h>
#include <pathtracer.cuh>
#include <mlt.cuh>

#include "cuda_runtime.h"

__device__ vec3 metropolis(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state) {

}