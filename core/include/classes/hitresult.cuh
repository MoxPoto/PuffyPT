#ifndef HITR_H
#define HITR_H

#include <classes/vec3.cuh>

struct HitResult {
	vec3 HitPos;
	vec3 HitNormal;
	float t, u, v;
	int objId;
	bool backface = false;

	__host__ __device__ HitResult() {};
};


#endif 