#include "mesh.cuh"
#include "triangle.cuh"
#include "vec3.cuh"
#include "object.cuh"
#include "hitresult.cuh"
#include "ray.cuh"

#include "cuda_runtime.h"
#include "stdio.h"
#include "dxhook/mainHook.h"
constexpr float kEpsilon = 1e-8;
constexpr float MAX_FLOAT = 1000000.0f;

#define MOLLER_TRUMBORE

__host__ __device__ static bool rayTriangleIntersect(
    const Tracer::vec3& orig, const Tracer::vec3& dir,
    const Tracer::vec3& v0, const Tracer::vec3& v1, const Tracer::vec3& v2,
    float& t, float& u, float& v)
{
#ifdef MOLLER_TRUMBORE 
    Tracer::vec3 v0v1 = v1 - v0;
    Tracer::vec3 v0v2 = v2 - v0;
    Tracer::vec3 pvec = cross(dir, v0v2);
    float det = (float)dot(v0v1, pvec);
#ifdef CULLING 
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return false;
#else 
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return false;
#endif 
    float invDet = 1 / det;

    Tracer::vec3 tvec = orig - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    Tracer::vec3 qvec = cross(tvec, v0v1);
    v = dot(dir, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    t = dot(v0v2, qvec) * invDet;

    return true;
#else 
    // compute plane's normal
    Tracer::vec3 v0v1 = v1 - v0;
    Tracer::vec3 v0v2 = v2 - v0;
    // no need to normalize
    Tracer::vec3 N = v0v1.cross(v0v2); // N 
    float denom = N.dot(N);

    // Step 1: finding P

    // check if ray and plane are parallel ?
    float NdotRayDirection = N.dot(dir);
    if (fabs(NdotRayDirection) < kEpsilon) // almost 0 
        return false; // they are parallel so they don't intersect ! 

    // compute d parameter using equation 2
    float d = N.dot(v0);

    // compute t (equation 3)
    t = (N.dot(orig) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind 

    // compute the intersection point using equation 1
    Tracer::vec3 P = orig + (dir * t);

    // Step 2: inside-outside test
    Tracer::vec3 C(0, 0, 0); // vector perpendicular to triangle's plane 

    // edge 0
    Tracer::vec3 edge0 = v1 - v0;
    Tracer::vec3  vp0 = P - v0;
    C = edge0.cross(vp0);
    if (N.dot(C) < 0) return false; // P is on the right side 

    // edge 1
    Tracer::vec3  edge1 = v2 - v1;
    Tracer::vec3  vp1 = P - v1;
    C = edge1.cross(vp1);
    if ((u = N.dot(C)) < 0)  return false; // P is on the right side 

    // edge 2
    Tracer::vec3  edge2 = v0 - v2;
    Tracer::vec3  vp2 = P - v2;
    C = edge2.cross(vp2);
    if ((v = N.dot(C)) < 0) return false; // P is on the right side; 

    u /= denom;
    v /= denom;

    return true; // this ray hits the triangle 
#endif 
}

#define MAX_TRIANGLES 9000
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )

namespace Tracer {
	__host__ __device__ Mesh::Mesh() {
		size = 0;
        checkCudaErrors(cudaMalloc((void**)&triBuffer, MAX_TRIANGLES * sizeof(Triangle)));
	}

    __host__ __device__ Mesh::~Mesh() {
        checkCudaErrors(cudaFree(triBuffer));
    }
	__host__ __device__ void Mesh::InsertTri(vec3 v1, vec3 v2, vec3 v3) {
		Triangle theTri(v1, v2, v3);

        if ((size + 1) >= MAX_TRIANGLES) {
            printf("MAX TRIANGLES LIMIT REACHED!!!!");
            return;
        }
        else {
            printf("[gpu]: Triangle inserted on GPU!");
        }

        *(triBuffer + size++) = theTri;
	}

    __host__ __device__ bool Mesh::tryHit(const Ray& ray, float closest, HitResult& result) {
        result.t = 0;
        result.u = 0;
        result.v = 0;

        float tMax = closest;
        bool didHit = false;

        for (int i = 0; i < size; i++) {
            Triangle triHere = *(triBuffer + i);
   

            if (rayTriangleIntersect(ray.origin, ray.direction, triHere.v1, triHere.v2, triHere.v3, result.t, result.u, result.v) && result.t < tMax) {
                tMax = result.t;
                result.HitPos = ray.origin + (ray.direction * result.t);
                result.HitNormal = triHere.normal;
                
                didHit = true;
            }
        }

        return didHit;
	}
}