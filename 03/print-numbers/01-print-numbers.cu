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
  for (int i = 0; i < 5; ++i)
  {
    printNumber << <1, 1 >> > (i);
  }
  cudaDeviceSynchronize();
}
