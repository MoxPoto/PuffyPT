#include "../object.cuh"
#include "../mesh.cuh"
#include "../vec3.cuh"
#include "../sphere.cuh"
#include "cuda_runtime.h"
#include "objects.cuh"
#include "../dxhook/mainHook.h"

#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

#define DEBUGHOST(str) printf("[host]: %s\n", str);
#define DEBUGGPU(str) printf("[gpu]: %s\n", str);

namespace Tracer {
	namespace CPU {
		__global__ void addObject(Tracer::Object** world, ObjectType obj_type, Pixel* texturePtr, int curCount) {
			switch (obj_type) {
			case (ObjectType::Sphere):
				printf("Adding sphere on GPU with curCount: %d, and obj_type: %d\n", curCount, obj_type);
				*(world + curCount) = (new Tracer::Sphere(vec3(0, 0, 0), 1.f));
				Tracer::Object* newObject = *(world + curCount);
				newObject->objectID = curCount;
				newObject->texture.Initialize(256, 256, texturePtr);

				DEBUGGPU("Finished sphere instantiation on GPU!");

				break;
			case (ObjectType::TriangleMesh):
				*(world + curCount) = (new Tracer::Mesh());
				Tracer::Object* newMesh = *(world + curCount);
				newMesh->objectID = curCount;
				newMesh->texture.Initialize(256, 256, texturePtr);

				break;
			default:
				break;
			}

			DEBUGGPU("[addObject]: Finished kernel, returning to host!");
		}

		int AddTracerObject(ObjectType type, Pixel* texturePtr) {
			DEBUGHOST("[host]: AddTracerObject called..");

			addObject << <1, 1 >> > (DXHook::world, type, texturePtr, DXHook::world_count);

			DEBUGHOST("[host]: Executed kernel..");
			checkCudaErrors(cudaGetLastError());
			DEBUGHOST("[host]: cudaGetLastError()");
			checkCudaErrors(cudaDeviceSynchronize());
			DEBUGHOST("[host]: cudaDeviceSynchronize()");

			int id = DXHook::world_count++;
			DEBUGHOST("[host]: id");

			return id;
		}

		__global__ void setObjectBRDF(Tracer::Object** world, BRDF type, int objectId) {
			Tracer::Object* target = *(world + objectId);

			target->matType = type;
		}

		CommandError SetBRDF(int objectId, BRDF type) {
			CommandError err = CommandError::Success;

			if (objectId >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			setObjectBRDF << <1, 1 >> > (DXHook::world, type, objectId);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			return err;
		}

		__global__ void setObjectClrEmission(Tracer::Object** world, vec3 color, float emission, int id) {
			Tracer::Object* target = *(world + id);

			target->color = color;
			target->emission = emission;
		}

		CommandError SetColorEmission(int objectID, vec3 color, float emission) {
			CommandError err = CommandError::Success;

			if (objectID >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			setObjectClrEmission << <1, 1 >> > (DXHook::world, color, emission, objectID);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			return err;
		}

		__global__ void copyObjectLighting(Tracer::Object** world, int id, LightingOptions* dest) {
			Tracer::Object* target = *(world + id);

			//cudaMemcpy(dest, &target->lighting, sizeof(LightingOptions), cudaMemcpyDeviceToHost);
		}

		CommandError GetObjectLighting(int id, LightingOptions* dest) {
			CommandError err = CommandError::Success;

			if (id >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			copyObjectLighting << <1, 1 >> > (DXHook::world, id, dest);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			return err;
		}

		__global__ void commitLightingOptions(Tracer::Object** world, int id, LightingOptions copy) {
			Tracer::Object* target = *(world + id);

			target->lighting = copy;
		}

		CommandError CommitObjectLighting(int id, LightingOptions copy) {
			CommandError err = CommandError::Success;

			if (id >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			commitLightingOptions << <1, 1 >> > (DXHook::world, id, copy);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			return err;
		}

		__global__ void insertCPUTri(Tracer::Object** world, int id, float u1, float u2, float u3, float vt1, float vt2, float vt3, float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3) {
			DEBUGGPU("Starting insertCPUTri");
			Tracer::Mesh* theMesh = reinterpret_cast<Tracer::Mesh*>(*(world + id));
			vec3 v1(x1, y1, z1);
			vec3 v2(x2, y2, z2);
			vec3 v3(x3, y3, z3);

			theMesh->InsertTri(v1, v2, v3, u1, u2, u3, vt1, vt2, vt3);
			DEBUGGPU("Inserted triangle (from what I see)");
		}

		CommandError InsertObjectTri(int id, vec3 v1, vec3 v2, vec3 v3, float u1, float u2, float u3, float vt1, float vt2, float vt3) {
			CommandError err = CommandError::Success;

			if (id >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			insertCPUTri << <1, 1 >> > (DXHook::world, id, u1, u2, u3, vt1, vt2, vt3, v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z(), v3.x(), v3.y(), v3.z());
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());


			DEBUGHOST("[InsertObjectTri]: Called!");

			return err;
		}

		__global__ void computeTriAccel(Tracer::Object** world, int id, vec3 nMin, vec3 nMax) {
			Tracer::Mesh* mesh = reinterpret_cast<Tracer::Mesh*>(*(world + id));

			mesh->ComputeAccel(nMin, nMax);
		}

		CommandError ComputeMeshAccel(int id, vec3 newMin, vec3 newMax) {
			CommandError err = CommandError::Success;

			if (id >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			computeTriAccel << <1, 1 >> > (DXHook::world, id, newMin, newMax);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			return err;
		}

		__global__ void setObjPosition(Tracer::Object** world, int id, vec3 newPos) {
			Tracer::Object* obj = *(world + id);

			obj->position = newPos;
		}

		CommandError SetObjectPosition(int id, vec3 position) {
			CommandError err = CommandError::Success;

			if (id >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			setObjPosition << <1, 1 >> > (DXHook::world, id, position);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			return err;
		}

		__global__ void setSphereSize(Tracer::Object** world, int id, float newSize) {
			Tracer::Sphere* ourSphere = static_cast<Tracer::Sphere*>(*(world + id));
			ourSphere->radius = newSize;
		}

		CommandError SetSphereSize(int id, float newSize) {
			CommandError err = CommandError::Success;

			if (id >= DXHook::world_count) {
				err = CommandError::NonexistantObject;
				return err;
			}

			setSphereSize << <1, 1 >> > (DXHook::world, id, newSize);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			return err;
		}

		void SetCameraPos(float x, float y, float z) {
			if (DXHook::curX != x || DXHook::curY != y || DXHook::curZ != z) {
				DXHook::frameCount = 0;
			}

			DXHook::curX = x;
			DXHook::curY = y;
			DXHook::curZ = z;
		}

		void SetCameraAngles(vec3 camDir) {
			if (DXHook::camDir.x() != camDir.x() || DXHook::camDir.y() != camDir.y() || DXHook::camDir.z() != camDir.z()) {
				DXHook::frameCount = 0;
			}

			DXHook::camDir = camDir;
		}
	}
}