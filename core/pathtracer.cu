#include <pathtracer.cuh>

#include "cuda_runtime.h"

#include <classes/vec3.cuh>
#include <classes/ray.cuh>
#include <classes/object.cuh>

#include <dxhook/mainHook.h>

#include <brdfs/lambert.cuh>
#include <brdfs/specular.cuh>
#include <brdfs/refraction.cuh>
#include <brdfs/mixed.cuh>

#include "math_constants.h"

#pragma region Utility


__device__ static inline vec3 lerpVectors(vec3 a, vec3 b, float f)
{
    return (a * (1.0f - f)) + (b * f);
}

__device__ static void adjustShadingNormal(HitResult& closestHit, vec3 dir) {
    vec3 Ng = closestHit.GeometricNormal;
    vec3 Ns = closestHit.HitNormal;

    const float kCosThetaThreshold = 0.1f;
    float cosTheta = dot(dir, Ns);

    if (cosTheta <= kCosThetaThreshold) {
        float t = __saturatef(cosTheta * (1.f / kCosThetaThreshold));
        closestHit.HitNormal = lerpVectors(Ng, Ns, t);
    }
}


__device__ Object* traceScene(int count, Object** world, const Ray& ray, HitResult& output, bool aabbOverride) {
    Object* hitObject = NULL;

    output.t = FLT_MAX;

    for (int i = 0; i < count; i++) {
        Object* target = *(world + i);

        if (i == ray.ignoreID) continue;

        if (target->AnyHit(ray)) {
            // ok, then we trace the precise mesh

            if (target->TryHit(ray, output)) {
                hitObject = target;
            }
        }
    }

    // Fix our shading normal and compute HitPos
    if (hitObject != NULL) {
        output.HitPos = ray.origin + (ray.direction * output.t);

        bool inverted = dot(ray.direction, output.GeometricNormal) > 0.f;
        output.backface = inverted;

        if (inverted) {
            output.HitNormal = -output.HitNormal;
            output.GeometricNormal = -output.GeometricNormal;
        }
        
        if (hitObject->pbrMaps.mraoMap.initialized) {
            output.MRAO = hitObject->pbrMaps.mraoMap.GetPixel(output.u, output.v);
        }

        output.HitAlbedo = hitObject->GetColor(output);

        if (hitObject->pbrMaps.emissionMap.initialized) {
            vec3 emissionColorHere = hitObject->pbrMaps.emissionMap.GetPixel(output.u, output.v);

            output.HitAlbedo += emissionColorHere * hitObject->emission;
        }
        
        // adjustShadingNormal(output, ray.direction);
    }

    return hitObject;
}

__device__ vec3 genSkyColor(HDRI* mainHDRI, SkyInfo skyInfo, float* imgData, const vec3& dir) {
    /*
    float t = 0.5f * (dir.z() + 1.0f);
    vec3 skyColor = (1.0f - t) * skyInfo.azimuth + t * skyInfo.zenith;
    */

    vec3 skyColor = mainHDRI->GetPixelFromRay(dir, imgData);

    return skyColor;
}

#pragma endregion Utility

#pragma region Shading

static __device__ const int EMISSIVE_MINIMUM = 15; // Minimum emission to be considered a light
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

__device__ vec3 calcDirect(int count, Object** world, Object* firstHit, const Ray& ray, const HitResult& rec) {


    vec3 lightObtained(0, 0, 0);
    int lightHits = 0;

    for (int i = 0; i < count; i++) {
        Object* light = *(world + i);

        if (light->emission >= EMISSIVE_MINIMUM) {
            float lightPower = 300.f + ((light->emission - EMISSIVE_MINIMUM) * 2.f); // The more intense emission is, more range is added
            float lightBrightness = 1.f;

            vec3 newOrigin = rec.HitPos + (rec.HitNormal * 0.001f);
            vec3 testDirection = unit_vector((light->position - newOrigin));

            Ray testRay(newOrigin, testDirection);
            HitResult testResult;

            Object* hitObject = traceScene(count, world, testRay, testResult);

            // A path from the sampled position and the light has been found
            if (hitObject != NULL && hitObject->objectID == light->objectID && testResult.t <= lightPower) {
                // float normalizedRange = (distance / lightPower);

                float falloff = lightPower / ((0.01 * 0.01) + powf(testResult.t, 2.f));

                vec3 lightContribution = (light->GetColor(testResult) * falloff) * lightBrightness;

                lightHits++;
                lightObtained += lightContribution;
            }
        }
    }

    if (lightHits == 0) {
        return lightObtained;
    }
    else {
        lightObtained /= static_cast<float>(lightHits);
        return lightObtained;
    }

}

static __device__ PathtraceResult depthColor(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state) {
    Ray cur_ray = ray;
    vec3 currentLight(1, 1, 1);
    
    PathtraceResult res;
    res.vertices = options->max_depth + 1;
    res.eyePath = reinterpret_cast<LightHit*>(malloc(sizeof(LightHit) * res.vertices));

    res.vertices = 0;

    for (int i = 0; i < options->max_depth; i++) {
        HitResult rec;
        Object* target = traceScene(options->count, options->world, cur_ray, rec);

        if (target != NULL) {
            // set our current ray to the new formulated one (this being perfect diffuse)
            // and attenuate our color by the albedo we hit, but we also should multiply our albedo by the objects emission

            if (!target->pbrMaps.emissionMap.initialized && target->emission > EMISSIVE_MINIMUM) {
                // just return the light
                LightHit hitPoint;
                hitPoint.hitPos = rec.HitPos;
                hitPoint.startPos = cur_ray.origin;

                hitPoint.attenuation = (target->GetColor(rec) * target->emission);
                hitPoint.pdf = 1.f;

                hitPoint.isLight = true;

                res.eyePath[res.vertices++] = hitPoint;
                res.color = currentLight * (target->GetColor(rec) * target->emission);
                return res;
            }

            Ray new_ray(vec3(0, 0, 0), vec3(0, 0, 0));
            vec3 attenuation = currentLight;
            float pdf = 1.f;

            /*
            switch (target->matType) {
            case (BRDF::Lambertian):
                LambertBRDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, attenuation, new_ray, target);
                break;
            case (BRDF::Specular):
                SpecularBRDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, cur_ray, attenuation, new_ray, target);
                break;
            case (BRDF::Refraction):
                RefractBRDF::SampleWorld(rec, local_rand_state, pdf, options->curtime, cur_ray, attenuation, new_ray, target);
                break;
            default:
                break;
            }
            */

            LightHit thisHit;
            thisHit.hitPos = rec.HitPos;
            thisHit.startPos = cur_ray.origin;
            thisHit.isLight = false;

            bool validSample = MixedBxDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, attenuation, cur_ray, new_ray, target, thisHit.brdf);

            if (!validSample) {
                // Nothing was chosen from our BxDF, so continue onwards
                continue;
            }

            thisHit.attenuation = attenuation;
            thisHit.pdf = pdf;

            currentLight *= attenuation / pdf;

            // russian roulette to terminate paths that barely contain any visible contribution
            // from: https://computergraphics.stackexchange.com/a/5808

            /*
            float prob = max(currentLight.x(), max(currentLight.y(), currentLight.z()));

            if (curand_uniform(local_rand_state) > prob) {
                return currentLight;
            }

            // ok, now we add the energy lost from russian rouletting:
            currentLight *= 1 / prob;
            */

            cur_ray = new_ray;

            res.eyePath[res.vertices++] = thisHit;
            
        }
        else {
            // didnt hit, finish our depth trace by attenuating our final hit color by the sky color

            if (options->doSky) {
                LightHit thisHit;
                thisHit.hitPos = rec.HitPos;
                thisHit.isLight = true;

                vec3 skyColor = genSkyColor(options->hdri, options->skyInfo, options->hdriData, cur_ray.direction);

                thisHit.attenuation = skyColor;
                thisHit.pdf = 1.f;
                
                res.color = (currentLight * (skyColor));
                res.eyePath[res.vertices++] = thisHit;

                return res;
            }
            else {
                LightHit thisHit;
                thisHit.hitPos = rec.HitPos;
                thisHit.isLight = true;
                
                const vec3 thisSkyColor = vec3(0.3, 0.3, 0.3);
                
                thisHit.attenuation = thisSkyColor;
                thisHit.pdf = 1.f;

                res.color = (currentLight * thisSkyColor);
                res.eyePath[res.vertices++] = thisHit;

                return res;
            }
        }
    }

    res.color = vec3(0.f, 0.f, 0.f);
    return res; // exceeded recursion..
}

__device__ PathtraceResult pathtrace(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state) {
    vec3 indirectLighting(0, 0, 0);
    vec3 directLighting(0, 0, 0);

    PathtraceResult res;

    HitResult result;
    Object* hitObject = traceScene(options->count, options->world, ray, result);

    if (hitObject != NULL) {
        directLighting = calcDirect(options->count, options->world, hitObject, ray, result);
    }

    for (int i = 0; i < options->samples; i++) {
        PathtraceResult depthRes = depthColor(options, ray, local_rand_state);
        indirectLighting += depthRes.color;
        res.eyePath = depthRes.eyePath;
        res.vertices = depthRes.vertices;
    }

    indirectLighting /= (float)options->samples;

    if (options->curPass == 0) { // Direct only
        res.color = directLighting;
    }
    else if (options->curPass == 1) { // Indirect only
        res.color = indirectLighting;
    }
    else {
        res.color = (directLighting / CUDART_PI + 2.0 * indirectLighting);
    }

    return res;
}

#pragma endregion Shading