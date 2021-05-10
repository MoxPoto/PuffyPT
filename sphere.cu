#include "sphere.cuh"

#include "vec3.cuh"
#include "object.cuh"
#include "hitresult.cuh"
#include "ray.cuh"

#include "cuda_runtime.h"
#include "stdio.h"

constexpr float FLOATE_MAX = 100000.f;

namespace Tracer {
    __host__ __device__ Sphere::Sphere(vec3 position, float radiuss) {
        radius = radiuss;
        center = position;
    }

    __device__ bool Sphere::tryHit(const Ray& ray, float closest, HitResult& result) {
        result.t = 0;
        result.u = 0;
        result.v = 0;

        float t_max = closest;
        float t_min = 0.001f;

        vec3 oc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0.f) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                result.t = temp;
                result.HitPos = ray.origin + (ray.direction * result.t);
                result.HitNormal = (result.HitPos - center) / radius;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                result.t = temp;
                result.HitPos = ray.origin + (ray.direction * result.t);
                result.HitNormal = (result.HitPos - center) / radius;
                return true;
            }
        }
        return false;
    }
}