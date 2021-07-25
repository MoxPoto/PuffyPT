#include <mlt/lightpaths.cuh>
#include <classes/vec3.cuh>
#include <pathtracer.cuh>

#include <brdfs/lambert.cuh>


namespace MLT {
	// Evaluates a light path and returns the color that represents it
	__device__ vec3 EvaluateLightPath(DXHook::RenderOptions* options, int vertices, LightHit* lightPath) {
		// First, we should check if the light path actually succeeded,
		// this means that we need to check if the last vertex is a light
        static const float EMISSIVE_MINIMUM = 15.f;

		bool didFinish = (lightPath[vertices - 1].isLight == true);
		// if it didn't just return 0, 0, 0

		if (!didFinish) {
			return vec3(0, 0, 0);
		}

		vec3 currentLight(1, 1, 1);
		Ray cur_ray;
		cur_ray.origin = lightPath[0].startPos;
		cur_ray.direction = lightPath[0].dir;

		for (int i = 1; i < vertices; i++) {
			HitResult rec;
			Object* target = traceScene(options->count, options->world, cur_ray, rec);

            if (target != NULL) {
                if (!target->pbrMaps.emissionMap.initialized && target->emission > EMISSIVE_MINIMUM) {
                    // just return the light
                    return currentLight * (target->GetColor(rec) * target->emission);
                }

                Ray new_ray(lightPath[i].startPos, lightPath[i].dir);

                vec3 attenuation = currentLight;
                float pdf = 1.f;

                /*
                    TODO:
                    add mixed bxdf support to MLT
                */
                /*
                bool validSample = MixedBxDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, attenuation, cur_ray, new_ray, target, thisHit.brdf);

                if (!validSample) {
                    // Nothing was chosen from our BxDF, so continue onwards
                    continue;
                }
                */

                LambertBRDF::Eval(rec.HitNormal, -cur_ray.direction, cur_ray.direction, rec.HitAlbedo, attenuation, pdf);

                currentLight *= attenuation / pdf;

                cur_ray = new_ray;

            }
            else {
                // didnt hit, finish our depth trace by attenuating our final hit color by the sky color

                if (options->doSky) {
                    vec3 skyColor = genSkyColor(options->hdri, options->skyInfo, options->hdriData, cur_ray.direction);

                    return (currentLight * (skyColor));
                }
                else {
                    const vec3 thisSkyColor = vec3(0.3, 0.3, 0.3);


                    return (currentLight * thisSkyColor);
                }
            }
		}
	
		// If we didn't hit a light path at all while traversing our vertices, then that means this light path was mutated unsucessfully..
		return vec3(0, 0, 0);
	}
}