#ifndef CG_OBJECTS_H
#define CG_OBJECTS_H

#include "../object.cuh"
#include "../mesh.cuh"
#include "../vec3.cuh"
#include "cuda_runtime.h"

// CPU to GPU interactions
// TODO: work on cpu gpu interaction
// firstly, world count must be organized, and a-
// object reflection thing should be worked on,
// personally I was thinking of each class adding their own kernels to modify
// which is a good idea so I avoid crazy shit like C++ reflection APIs
// anyways yeah, once this base is more thought out we need to work on services
// like making a "SynchronizationService" or some shit like that
// so I can fetch object positions from lua

namespace Tracer {
	namespace CPU {
		extern __global__ void addObject();
	}
}

#endif