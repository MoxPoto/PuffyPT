﻿#include "sphere.cuh"

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
        center = vec3(0, 0, 0); // deprecated in favor of the new position base class variable TODO: remove
    }

    __device__ bool Sphere::tryHit(const Ray& ray, HitResult& output) {
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
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                output.t = temp;
                output.HitPos = ray.origin + (ray.direction * output.t);
                output.HitNormal = (output.HitPos - position) / radius;
                return true;
            }
        }
        return false;
    }
}