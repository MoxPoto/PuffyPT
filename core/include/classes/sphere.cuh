#ifndef SPHERE_H
#define SPHERE_H

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/object.cuh>

class Sphere : public Object {
public:
	float radius = .2f;
	vec3 center = vec3(0, 0, 0);

	__host__ __device__ Sphere(vec3 position, float radius);
	__host__ __device__ bool virtual TryHit(const Ray& ray, HitResult& result);
};


#endif