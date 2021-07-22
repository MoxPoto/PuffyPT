#ifndef MLT_TYPES_H
#define MLT_TYPES_H

#include <pathtracer.cuh>
#include <classes/vec3.cuh>

namespace MLT {
	struct MLTPath {
		vec3 pixel;
		int vertices;
		LightHit* path;
	};
}

#endif