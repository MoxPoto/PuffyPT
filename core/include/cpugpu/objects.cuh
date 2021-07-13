#ifndef CG_OBJECTS_H
#define CG_OBJECTS_H

#include <classes/object.cuh>
#include <classes/mesh.cuh>
#include <classes/vec3.cuh>
#include "cuda_runtime.h"

#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

namespace CPU {
	enum ObjectType {
		TriangleMesh,
		SphereMesh
	};

	enum CommandError {
		NonexistantObject,
		Success
	};

	// Internal function to add a object into the GPU
	extern __global__ void addObject(Object** world, ObjectType obj_type, Pixel* texturePtr, int curCount);
	// Function that adds an object to the GPU and returns the ID of it
	extern int AddTracerObject(ObjectType type, Pixel* texturePtr);

	// Object-general functions
	extern __global__ void setObjectBRDF(Object** world, BRDF type, int id);
	extern CommandError SetBRDF(int objectID, BRDF type);

	extern __global__ void setObjectClrEmission(Object** world, vec3 color, float emission, int id);
	extern CommandError SetColorEmission(int objectID, vec3 color, float emission);

	extern __global__ void copyObjectLighting(Object** world, int id, LightingOptions* dest);
	extern CommandError GetObjectLighting(int id, LightingOptions* dest);

	extern __global__ void commitLightingOptions(Object** world, int id, LightingOptions copy);
	extern CommandError CommitObjectLighting(int id, LightingOptions copy);

	// Object-specific functions

	extern __global__ void insertCPUTri(Object** world, int id, TrianglePayload payload);
	extern CommandError InsertObjectTri(int id, TrianglePayload payload);

	extern __global__ void computeTriAccel(Object** world, int id, vec3 nMin, vec3 nMax);
	extern CommandError ComputeMeshAccel(int id, vec3 newMin, vec3 newMax);

	extern __global__ void setObjPosition(Object** world, int id, vec3 newPos);
	extern CommandError SetObjectPosition(int id, vec3 position);

	extern __global__ void setSphereSize(Object** world, int id, float newSize);
	extern CommandError SetSphereSize(int id, float newSize);

	extern __global__ void setPBR(Object** world, int id, int mraoX, int mraoY, Pixel* normal, Pixel* mrao);
	extern CommandError SetPBR(int id, int mraoX, int mraoY, Pixel* normal, Pixel* mrao);

	extern void SetCameraPos(float x, float y, float z);
	extern void SetCameraAngles(vec3 camDir);
}


#endif