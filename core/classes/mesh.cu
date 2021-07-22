#include <classes/mesh.cuh>
#include <classes/triangle.cuh>
#include <classes/vec3.cuh>
#include <classes/object.cuh>
#include <classes/hitresult.cuh>
#include <classes/ray.cuh>

#include "cuda_runtime.h"
#include "stdio.h"
#include <dxhook/mainHook.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>


constexpr float kEpsilon = 1e-8;
constexpr float MAX_FLOAT = 1000000.0f;

__host__ __device__ static bool rayTriangleIntersect(
    const vec3& orig, const vec3& dir,
    const vec3& v0, const vec3& v1, const vec3& v2,
    float& t, float& u, float& v)
{
    const vec3 edge1 = v1 - v0;
    const vec3 edge2 = v2 - v0;

    const vec3 h = cross(dir, edge2);
    const float a = dot(edge1, h);
    if (a > -kEpsilon && a < kEpsilon) return false;

    const float f = 1.f / a;
    const vec3 s = orig - v0;
    u = f * dot(s, h);
    if (u < 0 || u > 1) return false;

    const vec3 q = cross(s, edge1);
    v = f * dot(dir, q);
    if (v < 0 || u + v > 1) return false;

    t = f * dot(edge2, q);
    if (t > kEpsilon) return true;

    return false;
}

#define MAX_TRIANGLES 9000
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

__device__ bool done = false;

static inline void swap(float a, float b) {
    float temp = a;
    a = b;
    b = temp;
}


__host__ __device__ Mesh::Mesh() {
	size = 0;
    triBuffer = new Triangle*[MAX_TRIANGLES];
    minV = vec3(0, 0, 0);
    maxV = vec3(0, 0, 0);

}

__host__ __device__ Mesh::~Mesh() {
    for (int i = 0; i < size; i++) {
        delete triBuffer[i];
    }

    delete triBuffer;
}
__device__ void Mesh::InsertTri(const TrianglePayload& payload) {
    Triangle* theTri = new Triangle(payload);

    if ((size + 1) >= MAX_TRIANGLES) {
        printf("MAX TRIANGLES LIMIT REACHED!!!!");
        return;
    }

    triBuffer[size++] = theTri;
}

__host__ __device__ void Mesh::ComputeAccel(vec3 newMin, vec3 newMax) {
    // bounds[0] == min
    // bounds[1] == max
        
    minV = newMin;
    maxV = newMax;
        
       
    printf("min: %.2f, %.2f, %.2f\nmax: %.2f, %.2f, %.2f\n", minV.x(), minV.y(), minV.z(), maxV.x(), maxV.y(), maxV.z());
        
}

__device__ bool Mesh::AnyHit(const Ray& ray) {
        
    vec3 nLocal = ray.invorig - ray.invdir * (minV + maxV) / 2.f;

    vec3 k = vec3(abs(ray.invdir.x()), abs(ray.invdir.y()), abs(ray.invdir.z())) * (maxV - minV) / 2.f;
    vec3 t1 = -nLocal - k;
    vec3 t2 = -nLocal + k;

    double tNear = max(max(t1.x(), t1.y()), t1.z());
    double tFar = min(min(t2.x(), t2.y()), t2.z());


    return !(tNear > tFar || tFar < 0);
        

    /*
    vec3 tMin = (minV - ray.origin) / ray.direction;
    vec3 tMax = (maxV - ray.origin) / ray.direction;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x(), t1.y()), t1.z());
    float tFar = min(min(t2.x(), t2.y()), t2.z());

    tMaxA = tFar;

    return tNear > tFar;
    */

    //return true;
}

__device__ bool Mesh::TryHit(const Ray& ray, HitResult& closestHit) {
    bool bHit = false;
    Triangle* closestTri = nullptr;
    float finalU, finalV;

    for (int i = 0; i < size; i++) {
        Triangle* triHere = triBuffer[i];
   
        float t = 0.f;
        float u = 0.f;
        float v = 0.f;

        if (rayTriangleIntersect(ray.origin, ray.direction, triHere->v1, triHere->v2, triHere->v3, t, u, v) && t > kEpsilon && t < closestHit.t) {
            closestHit.t = t;
            closestHit.u = (1.f - u - v) * triHere->u1 + u * triHere->u2 + v * triHere->u3;
            closestHit.v = (1.f - u - v) * triHere->vt1 + u * triHere->vt2 + v * triHere->vt3;
                
            // Account for >1 and <1 UVs
            closestHit.u -= floorf(closestHit.u);
            closestHit.v -= floorf(closestHit.v);

            // Interpolate vertex normals (we are dealing with source assets.. pretty low poly)

            closestHit.HitNormal = (1.f - closestHit.u - closestHit.v) * triHere->n1 + closestHit.u * triHere->n2 + closestHit.v * triHere->n3;
            closestHit.GeometricNormal = closestHit.HitNormal;

            closestHit.objId = objectID;

            closestHit.RealU = u;
            closestHit.RealV = v;

            finalU = u;
            finalV = v;

            bHit = true;
            closestTri = triHere;
        }
    }

    if (closestTri != nullptr && bHit) {
        if (pbrMaps.normalMap.initialized) {
            // Proceed with normal mapping
            vec3 bitangent = (1.f - closestHit.u - closestHit.v) * closestTri->bin1 + closestHit.u * closestTri->bin2 + closestHit.v * closestTri->bin3;
            vec3 tangent = (1.f - closestHit.u - closestHit.v) * closestTri->tan1 + closestHit.u * closestTri->tan2 + closestHit.v * closestTri->tan2;

            glm::mat3 tbnMatrix(
                tangent.toGLM(),
                bitangent.toGLM(),
                closestHit.HitNormal.toGLM()
            );

            vec3 thisNormal = pbrMaps.normalMap.GetPixel(closestHit.u, closestHit.v) * 2.0f - 1.0;
            glm::vec3 worldSpaceNormal = tbnMatrix * thisNormal.toGLM();

            thisNormal = vec3(worldSpaceNormal.x, worldSpaceNormal.y, worldSpaceNormal.z);
            thisNormal.make_unit_vector();

            closestHit.HitNormal = thisNormal;
        }

        /*
        if (pbrMaps.mraoMap.initialized) {
            closestHit.MRAO = pbrMaps.mraoMap.GetPixel(closestHit.u, closestHit.v);
        }

        closestHit.HitAlbedo = GetColor(closestHit);

        if (pbrMaps.emissionMap.initialized) {
            vec3 emissionColorHere = pbrMaps.emissionMap.GetPixel(closestHit.u, closestHit.v);

            closestHit.HitAlbedo += emissionColorHere * emission;
        }
        */
    }

    return bHit;
}
