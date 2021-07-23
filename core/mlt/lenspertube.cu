#include <mlt/types.cuh>
#include <mlt/lenspertube.cuh>
#include <mlt/util.cuh>

#include <pathtracer.cuh>
#include <classes/vec3.cuh>
#include <curand_kernel.h>


namespace MLT {
	// Mutates X based on a lens pertubation, returns a boolean describing if it successfully mutated
	__device__ float LensPertubation(const MLTPath& X, MLTPath& Y, float widthOfImage, curandState* randState) {
		Y = X;

		float newX = Y.pixel.x();
		float newY = Y.pixel.y();

		const float quarterOfWidth = widthOfImage * .25;

		// See Appendix B for the reasoning of the r1 and r2 arguments,
		// https://www.researchgate.net/profile/Parris-Egbert/publication/228958110_A_practical_introduction_to_metropolis_light_transport/links/54a6f2380cf267bdb90a0724/A-practical-introduction-to-metropolis-light-transport.pdf?origin=publication_detail

		PixelOffset(1.f, quarterOfWidth, newX, newY, curand_uniform(randState), curand_uniform(randState));

		Y.pixel = vec3(newX, newY, 0);

		float pathDensity = 0.f;

		for (int i = 1; i < Y.vertices; i++) {

		}
	}
}