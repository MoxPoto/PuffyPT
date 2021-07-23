#ifndef MLT_LENSPERTUBE_H
#define MLT_LENSPERTUBE_H

#include <mlt/types.cuh>
#include <curand_kernel.h>
namespace MLT {
	// Mutates X based on a lens pertubation, returns a boolean describing if it successfully mutated
	extern float LensPertubation(const MLTPath& X, MLTPath& Y, float widthOfImage, curandState* randState);
}
#endif 