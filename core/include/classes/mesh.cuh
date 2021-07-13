#ifndef MESH_H
#define MESH_H

#include <classes/triangle.cuh>
#include <classes/vec3.cuh>
#include <classes/object.cuh>
#include <classes/hitresult.cuh>
#include <classes/ray.cuh>

class Mesh : public Object {
public:
	Triangle **triBuffer;
	int size;
	vec3 minV;
	vec3 maxV;

	__host__ __device__ Mesh();
	__host__ __device__ ~Mesh();
	__device__ void InsertTri(const TrianglePayload& payload);
	__host__ __device__ void Mesh::ComputeAccel(vec3 newMin, vec3 newMax);
	__device__ bool virtual TryHit(const Ray& ray, HitResult& closestHit);
	__device__ bool virtual AnyHit(const Ray& ray);
};


#endif