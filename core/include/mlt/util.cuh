#ifndef MLT_UTIL_H
#define MLT_UTIL_H

#include <classes/vec3.cuh>
// Utility functions from the appendix from:
// https://www.researchgate.net/profile/Parris-Egbert/publication/228958110_A_practical_introduction_to_metropolis_light_transport/links/54a6f2380cf267bdb90a0724/A-practical-introduction-to-metropolis-light-transport.pdf?origin=publication_detail

namespace MLT {
	extern vec3 PerturbVector(const vec3& dir, float randomVariable, float randomVariable2);
	extern void PixelOffset(float r1, float r2, float& x, float& y, float randVariable, float randVariable2);
}
#endif