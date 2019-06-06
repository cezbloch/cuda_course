#include <stdio.h>
#include <cstdlib>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  cudaStream_t stream[5];       // CUDA streams are of type `cudaStream_t`.

  for (int i = 0; i < 5; ++i)
  {
    cudaStreamCreate(&stream[i]); // Note that a pointer must be passed to `cudaCreateStream`.
  }

  for (int i = 0; i < 5; ++i)
  {
    printNumber << <1, 1, 0, stream[i] >> > (i);
  }

  cudaDeviceSynchronize();


  for (int i = 0; i < 5; ++i)
  {
    cudaStreamDestroy(stream[i]); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
  }
}
