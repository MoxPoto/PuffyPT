#include <classes/object.cuh>
#include <classes/ray.cuh>
#include <classes/hitresult.cuh>
#include <classes/vec3.cuh>


__host__ __device__ Object::Object() {
	color = vec3(0, 0, 0);
	emission = 1.f;
}

__device__ vec3 Object::GetColor(const HitResult& rayThatHit) {
		
	if (!texture.initialized)
		return color;

	return texture.GetPixel(rayThatHit.u, rayThatHit.v) * color;
		

	//return vec3(rayThatHit.u, rayThatHit.v, 1.f - rayThatHit.u - rayThatHit.v);
}

__device__ bool Object::TryHit(const Ray& ray, HitResult& result) {
	return false;
}

__device__ bool Object::AnyHit(const Ray& ray) {
	return true; // if something simply just returns "true" on the AnyHit pass it's pretty much safe to assume it's not accelerated
}
