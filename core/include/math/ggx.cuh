// GGX Definitions
#include "cuda_runtime.h"
#include <classes/vec3.cuh>

extern __device__ float thetaFromVec(vec3 vec);

// D(m)
extern __device__ float GGXDistribution(float width, float thetaM, const vec3& hitNormal, const vec3& microfacet);

// G1(v, m)
extern __device__ float GGXGeometry(const vec3& v, const vec3& n, const vec3& m, float width)