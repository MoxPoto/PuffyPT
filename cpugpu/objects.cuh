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
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

namespace Tracer {
	namespace CPU {
		enum ObjectType {
			TriangleMesh,
			Sphere
		};

		enum CommandError {
			NonexistantObject,
			Success
		};

		// Internal function to add a object into the GPU
		extern __global__ void addObject(Tracer::Object** world, ObjectType obj_type, int curCount);
		// Function that adds an object to the GPU and returns the ID of it
		extern int AddTracerObject(ObjectType type);

		// Object-general functions
		extern __global__ void setObjectBRDF(Tracer::Object** world, BRDF type, int id);
		extern CommandError SetBRDF(int objectID, BRDF type);

		extern __global__ void setObjectClrEmission(Tracer::Object** world, vec3 color, float emission, int id);
		extern CommandError SetColorEmission(int objectID, vec3 color, float emission);

		extern __global__ void copyObjectLighting(Tracer::Object** world, int id, LightingOptions* dest);
		extern CommandError GetObjectLighting(int id, LightingOptions* dest);

		extern __global__ void commitLightingOptions(Tracer::Object** world, int id, LightingOptions copy);
		extern CommandError CommitObjectLighting(int id, LightingOptions copy);

		// Object-specific functions

		extern __global__ void insertCPUTri(Tracer::Object** world, int id, float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3);
		extern CommandError InsertObjectTri(int id, vec3 v1, vec3 v2, vec3 v3);

		extern __global__ void computeTriAccel(Tracer::Object** world, int id, vec3 nMin, vec3 nMax);
		extern CommandError ComputeMeshAccel(int id, vec3 newMin, vec3 newMax);

		extern void SetCameraPos(float x, float y, float z);
		extern void SetCameraAngles(float pitch, float yaw, float roll);
	}
}

#endif