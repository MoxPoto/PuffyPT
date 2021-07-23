#include <mlt/lightpaths.cuh>
#include <classes/vec3.cuh>
#include <pathtracer.cuh>

namespace MLT {
	// Evaluates a light path and returns the color that represents it
	__device__ vec3 EvaluateLightPath(DXHook::RenderOptions* options, int vertices, LightHit* lightPath) {
		// First, we should check if the light path actually succeeded,
		// this means that we need to check if the last vertex is a light

		bool didFinish = (lightPath[vertices - 1].isLight == true);
		// if it didn't just return 0, 0, 0

		if (!didFinish) {
			return vec3(0, 0, 0);
		}

		vec3 currentLight(1, 1, 1);
		Ray cur_ray;
		cur_ray.origin = lightPath[0].startPos;
		cur_ray.direction = lightPath[0].dir;

		for (int i = 0; i < vertices; i++) {
			HitResult rec;
			Object* target = traceScene(options->count, options->world, cur_ray, rec);

			// Evaluate every single vertex in the path
			LightHit thisVertex = lightPath[i];
			/*
			if (thisVertex.isLight) {
				// lights dont have a pdf, which im pretty sure isnt correct..
				return currentLight * (thisVertex.attenuation);
			}

			currentLight *= (thisVertex.attenuation / thisVertex.pdf);
			*/
		}
	
		// If we didn't hit a light path at all while traversing our vertices, then that means this light path was mutated unsucessfully..
		return vec3(0, 0, 0);
	}
}