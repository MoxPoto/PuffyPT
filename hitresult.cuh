#ifndef HITR_H
#define HITR_H

#include "vec3.cuh"


namespace Tracer {
	class HitResult {
	public:
		vec3 HitPos;
		vec3 HitNormal;
		float t, u, v;
		int objId;

		__host__ __device__ HitResult();
	};
}

#endif 