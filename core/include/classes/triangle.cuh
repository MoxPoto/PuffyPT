#ifndef TRI_H
#define TRI_H

#include "cuda_runtime.h"
#include "math.h"
#include <classes/vec3.cuh>

struct TrianglePayload {
	vec3 v1;
	vec3 v2;
	vec3 v3;

	// normal mapping vectors
	vec3 bin1;
	vec3 bin2;
	vec3 bin3;

	vec3 tan1;
	vec3 tan2;
	vec3 tan3;

	vec3 n1;
	vec3 n2;
	vec3 n3;

	float u1;
	float u2;
	float u3;

	float vt1;
	float vt2;
	float vt3;
};


class Triangle {
public:
	vec3 v1;
	vec3 v2;
	vec3 v3;
	
	vec3 n1;
	vec3 n2;
	vec3 n3;

	vec3 bin1, bin2, bin3;
	vec3 tan1, tan2, tan3;

	float u1, u2, u3;
	float vt1, vt2, vt3; 

	__device__ Triangle(const TrianglePayload& payload);
	__device__ Triangle();
};


#endif