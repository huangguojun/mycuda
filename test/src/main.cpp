#include <iostream>

#ifdef USE_CUDA
#include "gpu.hpp"
#endif

int main()
{
    std::cout << "Hello, world!" << std::endl;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();
    SimpleDeviceCall();
    EnumGPU();
    JuliaGPU();
#endif

    return 0;
}
