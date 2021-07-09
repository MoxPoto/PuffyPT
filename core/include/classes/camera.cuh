#ifndef CAMERA_CUH
#define CAMERA_CUH
// this is a file with small structs.. so why not add unrelated stuff yea

#include <classes/vec3.cuh>

struct Camera {
	float exposure = 1.f;
};

struct SkyInfo {
	vec3 zenith = vec3(0.5, 0.7, 1.0); // stupid thing at the top
	vec3 azimuth = vec3(0, 0, 0); // stupid thing at the horizon
};


#endif