#include <images/hdri.cuh>
#include <iostream>
#include <vector>

extern __host__ bool CheckFileExists(const std::string& path);
extern __host__ bool LoadHDRI(const std::string& path);
extern __host__ void FindHDRIs(const std::string& folder, std::vector<std::string>& hdriList, int& finalArraySize);
