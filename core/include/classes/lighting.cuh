#ifndef LIGHTING_CUH
#define LIGHTING_CUH

#include <images/texture.cuh>

enum BRDF {
	Lambertian,
	Specular,
	Refraction
};

struct LightingOptions {
	float roughness = 1.0f;
	float ior = 1.5f;
	float transmission = 0.f;
	float metalness = 0.f;
};

// TODO:
// add futureproofing support for transmission maps
// aka, look for maps ending with "_transmission"
// there isnt any avaliable so its a low priority for now
struct PBRMap {
	Texture normalMap;
	Texture mraoMap;
	Texture emissionMap;
};

#endif