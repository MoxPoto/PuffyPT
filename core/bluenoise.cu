#include <bluenoise.cuh>
#include "cuda_runtime.h"
#include "math_constants.h"

#include <images/texture.cuh>
#include <classes/vec3.cuh>

#include <cpugpu/objects.cuh>
#include <vendor/stb_image.h>

static __device__ float fract(float number) {
    return number - floorf(number);
}

namespace Bluenoise {
    __device__ Texture* blueNoiseTex;
	__device__ const float goldenRatio = 1.618033988f;
	__device__ int frameNumber = 0;
    __device__ int MAX_FRAMES = 4096;
    __device__ bool initialized = false;

	// so simple!!
	__device__ vec3 CalculateSample(int sampleIndex, vec3 uv) {
        vec3 thisColor = blueNoiseTex->GetRawPixel(uv.x(), uv.y());
        
        vec3 thisSample = blueNoiseDisk[sampleIndex];

        float thisRotation = thisColor.r();

        // Before we continue, we need to animate the rotation based on time (so we can converge)

        float addFactor = ((frameNumber % MAX_FRAMES) * goldenRatio);
        thisRotation = fract(addFactor + thisRotation);

        float theta = thisRotation * 2.0 * static_cast<float>(CUDART_PI);

        float rotatedX = thisSample.x() * cosf(theta) - thisSample.y() * sinf(theta);
        float rotatedY = thisSample.x() * sinf(theta) - thisSample.y() * cosf(theta);

        vec3 finalSample(rotatedX, rotatedY);

        finalSample = finalSample * 0.5 + 0.5;

        return finalSample;
	}

    __device__ void Initialize() {
        blueNoiseTex = new Texture();
        
        vec3 blueNoiseDiskTemp[64] = {
            vec3(0.478712,0.875764),
            vec3(-0.337956,-0.793959),
            vec3(-0.955259,-0.028164),
            vec3(0.864527,0.325689),
            vec3(0.209342,-0.395657),
            vec3(-0.106779,0.672585),
            vec3(0.156213,0.235113),
            vec3(-0.413644,-0.082856),
            vec3(-0.415667,0.323909),
            vec3(0.141896,-0.939980),
            vec3(0.954932,-0.182516),
            vec3(-0.766184,0.410799),
            vec3(-0.434912,-0.458845),
            vec3(0.415242,-0.078724),
            vec3(0.728335,-0.491777),
            vec3(-0.058086,-0.066401),
            vec3(0.202990,0.686837),
            vec3(-0.808362,-0.556402),
            vec3(0.507386,-0.640839),
            vec3(-0.723494,-0.229240),
            vec3(0.489740,0.317826),
            vec3(-0.622663,0.765301),
            vec3(-0.010640,0.929347),
            vec3(0.663146,0.647618),
            vec3(-0.096674,-0.413835),
            vec3(0.525945,-0.321063),
            vec3(-0.122533,0.366019),
            vec3(0.195235,-0.687983),
            vec3(-0.563203,0.098748),
            vec3(0.418563,0.561335),
            vec3(-0.378595,0.800367),
            vec3(0.826922,0.001024),
            vec3(-0.085372,-0.766651),
            vec3(-0.921920,0.183673),
            vec3(-0.590008,-0.721799),
            vec3(0.167751,-0.164393),
            vec3(0.032961,-0.562530),
            vec3(0.632900,-0.107059),
            vec3(-0.464080,0.569669),
            vec3(-0.173676,-0.958758),
            vec3(-0.242648,-0.234303),
            vec3(-0.275362,0.157163),
            vec3(0.382295,-0.795131),
            vec3(0.562955,0.115562),
            vec3(0.190586,0.470121),
            vec3(0.770764,-0.297576),
            vec3(0.237281,0.931050),
            vec3(-0.666642,-0.455871),
            vec3(-0.905649,-0.298379),
            vec3(0.339520,0.157829),
            vec3(0.701438,-0.704100),
            vec3(-0.062758,0.160346),
            vec3(-0.220674,0.957141),
            vec3(0.642692,0.432706),
            vec3(-0.773390,-0.015272),
            vec3(-0.671467,0.246880),
            vec3(0.158051,0.062859),
            vec3(0.806009,0.527232),
            vec3(-0.057620,-0.247071),
            vec3(0.333436,-0.516710),
            vec3(-0.550658,-0.315773),
            vec3(-0.652078,0.589846),
            vec3(0.008818,0.530556),
            vec3(-0.210004,0.519896)
        };

        memcpy(blueNoiseDisk, blueNoiseDiskTemp, sizeof(vec3) * 64);
    }
	// Dump of blue noise samples..

    __device__ vec3 blueNoiseDisk[64];
}

__host__ bool LoadBluenoise(const char* path) {
    int width = 512;
    int height = 512;
    int channels = 3;
    int wanted = 3;

    Pixel* hostPtr = stbi_loadf(path, &width, &height, &channels, wanted);

    if (hostPtr == NULL) {
        return false; // Couldn't load bluenoise..
    }
    else {
        // Sick

        Pixel* devPtr = CreateTextureOnDevice(hostPtr, path, (static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(wanted) * sizeof(Pixel)));
        CPU::SetBluenoise(devPtr, width, height);

        return true;
    }

    return false;
}