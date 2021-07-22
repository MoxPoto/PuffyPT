#include <mlt/lightpaths.cuh>
#include <classes/vec3.cuh>
#include <pathtracer.cuh>

namespace MLT {
	// Evaluates a light path and returns the color that represents it
	vec3 EvaluateLightPath(int vertices, LightHit* lightPath) {
		// First, we should check if the light path actually succeeded,
		// this means that we need to check if the last vertex is a light

		bool didFinish = (lightPath[vertices - 1].isLight == true);
		// if it didn't just return 0, 0, 0

		if (!didFinish) {
			return vec3(0, 0, 0);
		}

		vec3 currentLight(1, 1, 1);

		for (int i = 0; i < vertices; i++) {
			// Evaluate every single vertex in the path
			LightHit thisVertex = lightPath[i];

			if (thisVertex.isLight) {
				// lights dont have a pdf, which im pretty sure isnt correct..
				return currentLight * (thisVertex.attenuation);
			}

			currentLight *= (thisVertex.attenuation / thisVertex.pdf);
		}
	
		// If we didn't hit a light path at all while traversing our vertices, then that means this light path was mutated unsucessfully..
		return vec3(0, 0, 0);
	}
}