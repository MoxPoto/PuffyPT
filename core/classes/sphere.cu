#include <classes/sphere.cuh>

#include <classes/vec3.cuh>
#include <classes/object.cuh>
#include <classes/hitresult.cuh>
#include <classes/ray.cuh>

#include "cuda_runtime.h"
#include "stdio.h"
#include "math_constants.h"

__host__ __device__ Sphere::Sphere(vec3 position, float radiuss) {
    radius = radiuss; 
    center = vec3(0, 0, 0); // deprecated in favor of the new position base class variable TODO: remove
}

__device__ bool Sphere::TryHit(const Ray& ray, HitResult& output) {
    float t_max = output.t;
    float t_min = 0.001f;

    vec3 oc = ray.origin - position;
    float a = dot(ray.direction, ray.direction);
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0.f) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            output.t = temp;
            output.HitPos = ray.origin + (ray.direction * output.t);
            output.HitNormal = (output.HitPos - position) / radius;
            output.GeometricNormal = output.HitNormal;

            output.u = (1 + atan2(output.HitNormal.y(), output.HitNormal.x()) / CUDART_PI) * 0.5;
            output.v = acosf(output.HitNormal.z()) / CUDART_PI;

            output.RealU = output.u;
            output.RealV = output.v;

            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            output.t = temp;
            output.HitPos = ray.origin + (ray.direction * output.t);
            output.HitNormal = (output.HitPos - position) / radius;
            output.GeometricNormal = output.HitNormal;

            output.u = (1 + atan2(output.HitNormal.y(), output.HitNormal.x()) / CUDART_PI) * 0.5;
            output.v = acosf(output.HitNormal.z()) / CUDART_PI;

            output.RealU = output.u;
            output.RealV = output.v;

            return true;
        }
    }
    return false;
}
