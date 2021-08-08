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

#include <flags/montecarlo.cuh>
#include <bluenoise.cuh>

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

__device__ vec3 calcDirect(DXHook::RenderOptions* options, const HitResult& rec, curandState* local_rand_state) {
    Object* lightObjects[100];
    // Aux array to find light objects
    // TODO: MAKE THIS BE HANDLED BY THE MESH UPLOADER!!!

    int lightCount = 0;

    for (int i = 0; i < options->count; i++) {
        if (lightCount + 1 >= 100) {
            break;
        }

        Object* potentialLight = options->world[i];

        if (potentialLight->emission >= EMISSIVE_MINIMUM) {
            lightObjects[lightCount++] = potentialLight;
        }
    }

    if (lightCount <= 0)
        return vec3(0, 0, 0);

    float lightSelectionPDF = 1.f / lightCount;

    float rand = curand_uniform(local_rand_state);
    int idx = static_cast<int>(floorf(rand * (lightCount - 1)));

    Object* chosenLight = lightObjects[idx];

    vec3 lightDir = unit_vector(chosenLight->position - rec.HitPos);
    vec3 startPos = rec.HitPos + (rec.HitNormal * 0.01f);

    float lightPower = 60.f + ((chosenLight->emission - EMISSIVE_MINIMUM) * 2.f);

    HitResult testResult;

    Ray newRay;
    newRay.origin = startPos;
    newRay.direction = lightDir;

    Object* hitObject = traceScene(options->count, options->world, newRay, testResult);

    if (hitObject != NULL && hitObject->objectID == chosenLight->objectID && testResult.t <= lightPower) {
        float falloff = lightPower / ((0.01 * 0.01) + powf(testResult.t, 2.f));

        vec3 lightContrib = (chosenLight->GetColor(testResult) * falloff);

        return lightContrib / lightSelectionPDF;
    }

    return vec3(0, 0, 0);
}

static __device__ PathtraceResult depthColor(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state, vec3 thisUV) {
    Ray cur_ray = ray;
    vec3 currentLight(1, 1, 1);

    /*
    (directLighting / CUDART_PI + 2.0 * indirectLighting);  
    */
    
    PathtraceResult res;
    res.vertices = options->max_depth + 1;
#ifdef DO_MLT
    res.eyePath = reinterpret_cast<LightHit*>(malloc(sizeof(LightHit) * res.vertices));
#endif

    res.vertices = 0;

    BRDF lastBRDF = BRDF::Lambertian;
    res.specularOverride = false;

    for (int i = 0; i < options->max_depth; i++) {
        HitResult rec;

        Object* target = traceScene(options->count, options->world, cur_ray, rec);

        if (target != NULL) {
            // set our current ray to the new formulated one (this being perfect diffuse)
            // and attenuate our color by the albedo we hit, but we also should multiply our albedo by the objects emission

            if (!target->pbrMaps.emissionMap.initialized && target->emission > EMISSIVE_MINIMUM) {
                // just return the light
                LightHit hitPoint;
                hitPoint.hitResult = rec;
                hitPoint.startPos = cur_ray.origin;
                hitPoint.hitEntity = target;
                hitPoint.dir = cur_ray.direction;

                hitPoint.isLight = true;
#ifdef DO_MLT
                res.eyePath[res.vertices++] = hitPoint;
#endif
                res.color = currentLight * (target->GetColor(rec) * target->emission);
                return res;
            }
            else if (target->pbrMaps.emissionMap.initialized && target->emission > EMISSIVE_MINIMUM) {
                vec3 lightColor = (target->pbrMaps.emissionMap.GetPixel(rec.u, rec.v) * target->emission);

                if (lightColor.squared_length() > 0) {
                    // just return the light
                    LightHit hitPoint;
                    hitPoint.hitResult = rec;
                    hitPoint.startPos = cur_ray.origin;
                    hitPoint.hitEntity = target;
                    hitPoint.dir = cur_ray.direction;

                    hitPoint.isLight = true;
#ifdef DO_MLT
                    res.eyePath[res.vertices++] = hitPoint;
#endif
                    res.color = currentLight * lightColor;
                    return res;
                }
            }

            Ray new_ray(vec3(0, 0, 0), vec3(0, 0, 0));
            vec3 attenuation = currentLight;
            float pdf = 1.f;

            LightHit thisHit;
            thisHit.hitResult = rec;
            thisHit.startPos = cur_ray.origin;
            thisHit.isLight = false;
            thisHit.hitEntity = target;
            thisHit.dir = cur_ray.direction;

            if (Flags::estimatorType == Flags::MonteCarlo::Normal) {
                if (target->lighting.transmission <= 0.0f) {
                    bool validSample = MixedBxDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, attenuation, cur_ray, new_ray, target, thisHit.brdf, thisUV, i);

                    if (!validSample) {
                        // Nothing was chosen from our BxDF, so continue onwards
                        continue;
                    }
                }
                else {
                    // quick patch in for refraction
                    RefractBRDF::SampleWorld(rec, local_rand_state, pdf, options->curtime, cur_ray, attenuation, new_ray, target, thisHit.brdf);
                }
            }
            else if (Flags::estimatorType == Flags::MonteCarlo::Quasi) {
                LambertBRDF::SampleWorld(rec, local_rand_state, options->curtime, pdf, attenuation, new_ray, target, thisUV, i);
            }

            if (lastBRDF == BRDF::Specular) {
                // override the gbuffer with new data, BUT, we can't replace the diffuse color

                res.specularOverride = true;

                res.gbufferOverride.albedo = target->GetColor(rec);
                res.gbufferOverride.depth = fabsf((cur_ray.origin - rec.HitPos).length());
                res.gbufferOverride.normal = rec.HitNormal;
                res.gbufferOverride.brdfType = thisHit.brdf;
                res.gbufferOverride.isSky = false;
                res.gbufferOverride.objectID = target->objectID;
                res.gbufferOverride.position = rec.HitPos;
                
 
            } 

            vec3 indirect = (attenuation / pdf);
            vec3 direct = calcDirect(options, rec, local_rand_state);

            vec3 combined = (direct / CUDART_PI + 2.0 * indirect);

            currentLight *= indirect;

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
            lastBRDF = thisHit.brdf;
#ifdef DO_MLT
            res.eyePath[res.vertices++] = thisHit;
#endif
        }
        else {
            // didnt hit, finish our depth trace by attenuating our final hit color by the sky color
            res.gbufferOverride.isSky = true;

            if (options->doSky) {
                LightHit thisHit;
                thisHit.hitResult = rec;
                thisHit.startPos = cur_ray.origin;
               
                thisHit.isLight = true;
                thisHit.hitEntity = target;

                thisHit.dir = cur_ray.direction;

                vec3 skyColor = genSkyColor(options->hdri, options->skyInfo, options->hdriData, cur_ray.direction);

                res.color = (currentLight * (skyColor));
#ifdef DO_MLT
                res.eyePath[res.vertices++] = thisHit;
#endif
                return res;
            }
            else {
                LightHit thisHit;
                thisHit.hitResult = rec;
                thisHit.startPos = cur_ray.origin;

                thisHit.isLight = true;
                
                thisHit.hitEntity = target;
                thisHit.dir = cur_ray.direction;
                const vec3 thisSkyColor = vec3(0.3, 0.3, 0.3);
                

                res.color = (currentLight * thisSkyColor);
#ifdef DO_MLT
                res.eyePath[res.vertices++] = thisHit;
#endif
                return res;
            }

        }
    }

    res.color = vec3(0.f, 0.f, 0.f);
    return res; // exceeded recursion..
}

__device__ PathtraceResult pathtrace(DXHook::RenderOptions* options, const Ray& ray, curandState* local_rand_state, int x, int y) {
    vec3 indirectLighting(0, 0, 0);
    vec3 directLighting(0, 0, 0);

    vec3 thisUV(x, y);

    PathtraceResult res;

    HitResult result;
    Object* hitObject = traceScene(options->count, options->world, ray, result);

    for (int i = 0; i < options->samples; i++) {
        PathtraceResult depthRes = depthColor(options, ray, local_rand_state, thisUV);
        indirectLighting += depthRes.color;
        res.eyePath = depthRes.eyePath;
        res.vertices = depthRes.vertices;
        res.specularOverride = depthRes.specularOverride;
        res.gbufferOverride = depthRes.gbufferOverride;
    }

    indirectLighting /= (float)options->samples;

    if (options->curPass == 0) { // Direct only
        res.color = vec3(0, 0, 0);
    }
    else if (options->curPass == 1) { // Indirect only
        res.color = indirectLighting;
    }
    else {
        res.color = indirectLighting;
    }

    return res;
}

#pragma endregion Shading