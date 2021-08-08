#include <flags/montecarlo.cuh>

#include "cuda_runtime.h"

namespace Flags {
	__device__ MonteCarlo estimatorType = MonteCarlo::Normal;
}
