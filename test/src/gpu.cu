#include <iostream>
#include "gpu.hpp"

void printCudaVersion()
{
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}



__device__ int addem( int a, int b ) 
{
    return a + b;
}

__global__ void add( int a, int b, int *c ) 
{
    *c = addem( a, b );
}

void SimpleDeviceCall()
{

    int c;
    int *dev_c;
    cudaMalloc( (void**)&dev_c, sizeof(int) );

    add<<<1,1>>>( 2, 7, dev_c );

    cudaMemcpy( &c, dev_c, sizeof(int),cudaMemcpyDeviceToHost );
    printf( "2 + 7 = %d\n", c );

    cudaFree( dev_c ) ;
}


