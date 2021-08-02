// Monte-Carlo estimator flags

#ifndef FLAGS_MONTE_CARLO_H
#define FLAGS_MONTE_CARLO_H

#include "cuda_runtime.h"

namespace Flags {
	__device__ enum class MonteCarlo : unsigned char {
		Normal,
		Quasi
	};
	
	__device__ MonteCarlo estimatorType = MonteCarlo::Normal;
}

#endif