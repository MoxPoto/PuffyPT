#ifndef MESH_H
#define MESH_H

#include "triangle.cuh"
#include "vec3.cuh"
#include "object.cuh"
#include "hitresult.cuh"
#include "ray.cuh"


namespace Tracer {
	class Mesh : public Object {
	public:
		Triangle **triBuffer;
		int size;
		vec3 minV;
		vec3 maxV;

		__host__ __device__ Mesh();
		__host__ __device__ ~Mesh();
		__device__ void InsertTri(vec3 v1, vec3 v2, vec3 v3, float u1, float u2, float u3, float vt1, float vt2, float vt3);
		__host__ __device__ void Mesh::ComputeAccel(vec3 newMin, vec3 newMax);
		__host__ __device__ bool virtual tryHit(const Ray& ray, HitResult& closestHit);
		__host__ __device__ bool virtual anyHit(const Ray& ray);
	};
}

#endif