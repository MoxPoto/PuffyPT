﻿#include <classes/triangle.cuh>
#include <classes/vec3.cuh>

#include "cuda_runtime.h"
#include "math.h"

__device__ Triangle::Triangle(const TrianglePayload& payload) {
	v1 = payload.v1, v2 = payload.v2, v3 = payload.v3;
	u1 = payload.u1, u2 = payload.u2, u3 = payload.u3;
	vt1 = payload.vt1, vt2 = payload.vt2, vt3 = payload.vt3;

	vec3 theU = (v2 - v1);
	vec3 theV = (v3 - v1);
		
	normal = cross(theV, theU);
	normal.make_unit_vector(); // wtf

}

__device__ Triangle::Triangle() {
	v1 = vec3();
	v2 = vec3();
	v3 = vec3();

	u1 = 0.0f;
	u2 = 0.0f;
	u3 = 0.0f;

	vt1 = 0.0f;
	vt2 = 0.0f;
	vt3 = 0.0f;

	normal = vec3();
}
