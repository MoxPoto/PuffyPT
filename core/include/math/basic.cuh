// Localization stuff

#include <classes/vec3.cuh>

extern __device__ vec3 TransformToWorld(const float& x, const float& y, const float& z, const vec3& normal);
extern __device__ inline vec3 lerpVectors(vec3 a, vec3 b, float f);
extern __device__ float clamp(float d, float min, float max);
extern __device__ inline float sign(const float& value);