#pragma once
#include <chrono>

cudaEvent_t start, stop;
std::chrono::time_point<std::chrono::high_resolution_clock> cpu_start, cpu_stop;

inline void StartGpuTimer() 
{
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
};

inline void StartCpuTimer()
{
  cpu_start = std::chrono::high_resolution_clock::now();
};

inline double GetGpuTimerInMiliseconds() 
{ 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedMiliseconds;
  cudaEventElapsedTime(&elapsedMiliseconds, start, stop);

  return elapsedMiliseconds;
};

inline double GetCpuTimerInMiliseconds()
{
  cpu_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = cpu_stop - cpu_start;
  return diff.count();
};