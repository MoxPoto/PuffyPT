#include <classes/vec3.cuh>
#include <math/basic.cuh>

__device__ inline float sign(const float& value) {
	if (value < 0.f) return -1.f;

	return 1.f;
}

__device__ vec3 TransformToWorld(const float& x, const float& y, const float& z, const vec3& normal)
{
	// Find an axis that is not parallel to normal
	vec3 majorAxis;
	if (fabsf(normal.x()) < 0.57735026919f /* 1 / sqrt(3) */) {
		majorAxis = vec3(1, 0, 0);
	}
	else if (fabsf(normal.y()) < 0.57735026919f /* 1 / sqrt(3) */) {
		majorAxis = vec3(0, 1, 0);
	}
	else {
		majorAxis = vec3(0, 0, 1);
	}

	// Use majorAxis to create a coordinate system relative to world space
	vec3 u = unit_vector(cross(normal, majorAxis));
	vec3 v = cross(normal, u);
	vec3 w = normal;

	// Transform from local coordinates to world coordinates
	return u * x + v * y + w * z;
}

__device__ inline vec3 lerpVectors(vec3 a, vec3 b, float f)
{
	return (a * (1.0f - f)) + (b * f);
}

__device__ float clamp(float d, float min, float max) {
	const float t = d < min ? min : d;
	return t > max ? max : t;
}
