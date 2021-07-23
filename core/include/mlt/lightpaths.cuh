#ifndef MLT_LIGHTPATH_H
#define MLT_LIGHTPATH_H

#include <classes/vec3.cuh>
#include <pathtracer.cuh>
#include <mlt/types.cuh>

#include <dxhook/mainHook.h>

// Definitions for the MLT light path evaluator
// it takes in a light path, and generates an output color that matches what the light path stores

namespace MLT {
	__device__ extern vec3 EvaluateLightPath(DXHook::RenderOptions* options, int vertices, LightHit* lightPath);
}
#endif