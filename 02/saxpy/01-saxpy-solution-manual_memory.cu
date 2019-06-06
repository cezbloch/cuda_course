#include <stdio.h>
#include <cstdlib>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nvprof to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < N; i += stride)
  {
	  c[i] = 2 * a[i] + b[i];
  }    
}

int main()
{
  int *a, *b, *c, *h_a, *h_b, *h_c;

  int size = N * sizeof(int); // The total number of bytes per vector

  cudaMalloc(&a, size);
  cudaMalloc(&b, size);
  cudaMalloc(&c, size);

  cudaMallocHost(&h_a, size);
  cudaMallocHost(&h_b, size);
  cudaMallocHost(&h_c, size);

  // Initialize memory
  for (int i = 0; i < N; ++i)
  {
    h_a[i] = 2;
    h_b[i] = 1;
    h_c[i] = 0;
  }

  cudaMemcpy(a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b, h_b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(c, h_c, size, cudaMemcpyHostToDevice);

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  int threads_per_block = 512;
  int number_of_blocks = props.multiProcessorCount * 512;
  //int number_of_blocks = N / threads_per_block + 1;

  saxpy << < number_of_blocks, threads_per_block >> > (a, b, c);

  auto syncErr = cudaGetLastError();
  auto asyncErr = cudaDeviceSynchronize();

  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaMemcpy(h_c, c, size, cudaMemcpyDeviceToHost);

  // Print out the first and last 5 values of c for a quality check
  for (int i = 0; i < 5; ++i)
    printf("c[%d] = %d, ", i, h_c[i]);
  printf("\n");
  for (int i = N - 5; i < N; ++i)
    printf("c[%d] = %d, ", i, h_c[i]);
  printf("\n");

  cudaFree(a); cudaFree(b); cudaFree(c);
  cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
}
