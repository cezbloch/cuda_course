#include <stdio.h>
#include <cstdlib>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


void initWith(float num, float *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addArraysInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for (int i = 0; i < N; i++)
  {
    if (array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);

  const int N = 2 << 24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;
  float *h_a;
  float *h_b;
  float *h_c;

  cudaMalloc(&a, size);
  cudaMalloc(&b, size);
  cudaMalloc(&c, size);
  cudaMallocHost(&h_a, size);
  cudaMallocHost(&h_b, size);
  cudaMallocHost(&h_c, size);

  int threadsPerBlock;
  int numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addArraysErr;
  cudaError_t asyncErr;

  initWith(3, h_a, N);
  initWith(4, h_b, N);
  initWith(0, h_c, N);

  cudaMemcpy(a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b, h_b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(c, h_c, size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  addArraysInto << <numberOfBlocks, threadsPerBlock >> > (c, a, b, N);

  cudaMemcpy(h_c, c, size, cudaMemcpyDeviceToHost);

  addArraysErr = cudaGetLastError();
  if (addArraysErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addArraysErr));

  asyncErr = cudaDeviceSynchronize();
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, h_c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFreeHost(h_c);
}
