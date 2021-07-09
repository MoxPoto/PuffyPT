#include "cuda_runtime.h"
#include <images/hdri.cuh>
#include <images/hdriUtility.cuh>

#include <iostream>
#include <vector>
#include <filesystem>

#include <vendor/stb_image.h>
#include <util/macros.h>
#include <dxhook/mainHook.h>

// Cuda doesn't support C++17, so a seperate host compilation is required

__host__ void FindHDRIs(const std::string& folder, std::vector<std::string>& hdriList, int& finalArraySize) {
    int size = 0;
    hdriList.clear();

    for (const auto& hdri : std::filesystem::directory_iterator(folder)) {
        std::string foundPath = hdri.path().string();
        hdriList.push_back(foundPath);
    }

    // C++17 is so nice
    finalArraySize = hdriList.size();
}

