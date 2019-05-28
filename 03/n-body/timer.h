#pragma once

cudaEvent_t start, stop;


inline void StartTimer() 
{
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

};

inline double GetTimer() 
{ 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  return elapsedTime;
};