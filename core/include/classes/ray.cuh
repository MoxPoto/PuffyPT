#ifndef RAY_H
#define RAY_H

#include "cuda_runtime.h"
#include <classes/vec3.cuh>

struct Ray {
	vec3 origin;
	vec3 direction;
	vec3 invdir;
	vec3 invorig;

	int ignoreID;
	int sign[3];

	// is it just me or do these initializer lists look so cursed
	__host__ __device__ Ray(vec3 orig = vec3(0, 0, 0), vec3 dir = vec3(0, 0, 0), int ignore = -1) : origin(orig), direction(dir), ignoreID(ignore) {
		invdir = vec3(1.f, 1.f, 1.f) / dir;
		invorig = origin * invdir;

		sign[0] = (invdir.x() < 0);
		sign[1] = (invdir.y() < 0);
		sign[2] = (invdir.z() < 0);

		// this is used for the ray-box intersection tutorial at scratchapixel: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
	};
};


#endif