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

		__host__ __device__ Mesh();
		__host__ __device__ ~Mesh();
		__host__ __device__ void InsertTri(vec3 v1, vec3 v2, vec3 v3);
		__host__ __device__ bool virtual tryHit(const Ray& ray, float closest, HitResult& result);
	};
}

#endif