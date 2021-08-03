#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "cuda_runtime.h"

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/object.cuh>

#include <dxhook/mainHook.h>
#include <postprocess/mainDenoiser.cuh>

// #define DO_MLT

extern __device__ Object* traceScene(int count, Object** world, const Ray& ray, HitResult& output, bool aabbOverride = false);

struct LightHit {
	// Lighting information
	BRDF brdf;
	bool isLight;

	// Hit information (a copy of the hitresult)
	vec3 startPos;
	vec3 dir;

	HitResult hitResult;
	Object* hitEntity;
};

struct PathtraceResult {
	vec3 color;

	// This is meant for things like reflections, where the HIT normal isn't necessarily the normal that is
	// presented to the user,
	// this also helps immensely with denoising reflections, including rough ones

	bool specularOverride;
	Post::GBuffer gbufferOverride;

	int vertices;
	LightHit* eyePath;
};

extern __device__ vec3 genSkyColor(HDRI* mainHDRI, SkyInfo skyInfo, float* imgData, const vec3& dir);

extern __device__ PathtraceResult pathtrace(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state, int x, int y);

#endif