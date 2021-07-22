#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "cuda_runtime.h"

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/object.cuh>

#include <dxhook/mainHook.h>


extern __device__ Object* traceScene(int count, Object** world, const Ray& ray, HitResult& output, bool aabbOverride = false);

struct LightHit {
	// Lighting information
	BRDF brdf;
	vec3 attenuation;
	float pdf = 1.f;
	bool isLight;

	// Hit information
	vec3 hitPos;
	vec3 startPos;
};

struct PathtraceResult {
	vec3 color;
	int vertices;
	LightHit* eyePath;
};

extern __device__ vec3 genSkyColor(HDRI* mainHDRI, SkyInfo skyInfo, float* imgData, const vec3& dir);

extern __device__ PathtraceResult pathtrace(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state);

#endif