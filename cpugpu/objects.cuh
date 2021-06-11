#ifndef CG_OBJECTS_H
#define CG_OBJECTS_H

#include "../object.cuh"
#include "../mesh.cuh"
#include "../vec3.cuh"
#include "cuda_runtime.h"

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

		extern __global__ void insertCPUTri(Tracer::Object** world, int id, float u1, float u2, float u3, float vt1, float vt2, float vt3, float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3);
		extern CommandError InsertObjectTri(int id, vec3 v1, vec3 v2, vec3 v3, float u1, float u2, float u3, float vt1, float vt2, float vt3);

		extern __global__ void computeTriAccel(Tracer::Object** world, int id, vec3 nMin, vec3 nMax);
		extern CommandError ComputeMeshAccel(int id, vec3 newMin, vec3 newMax);

		extern __global__ void setObjPosition(Tracer::Object** world, int id, vec3 newPos);
		extern CommandError SetObjectPosition(int id, vec3 position);

		extern __global__ void setSphereSize(Tracer::Object** world, int id, float newSize);
		extern CommandError SetSphereSize(int id, float newSize);

		extern void SetCameraPos(float x, float y, float z);
		extern void SetCameraAngles(vec3 camDir);
	}
}

#endif