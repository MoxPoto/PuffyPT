// GGX Definitions
#include "cuda_runtime.h"
#include <classes/vec3.cuh>

extern __device__ float thetaFromVec(vec3 vec);

// D(m)
extern __device__ float GGXDistribution(float width, float thetaM, const vec3& hitNormal, const vec3& microfacet);

// G(i, o, m)
extern __device__ float GGXGeometry(const vec3& i, const vec3& o, const vec3& m, const vec3& n, float width);