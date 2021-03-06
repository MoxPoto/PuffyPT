// Localization stuff

#include <classes/vec3.cuh>


extern __device__ vec3 reflect(const vec3& direction, const vec3& normal);
extern __device__ float schlick(float cosine, float ref_idx);

extern __device__ vec3 TransformToWorld(const float& x, const float& y, const float& z, const vec3& normal);
extern __device__ vec3 lerpVectors(vec3 a, vec3 b, float f);
extern __device__ float clamp(float d, float min, float max);
extern __device__ float sign(const float& value);