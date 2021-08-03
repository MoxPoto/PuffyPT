// Monte-Carlo estimator flags

#ifndef FLAGS_MONTE_CARLO_H
#define FLAGS_MONTE_CARLO_H

#include "cuda_runtime.h"

namespace Flags {
	__host__ __device__ enum class MonteCarlo : unsigned char {
		Normal,
		Quasi
	};
	
	extern __device__ MonteCarlo estimatorType;
}

#endif