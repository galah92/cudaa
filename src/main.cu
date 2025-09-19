#include <cuda_runtime.h>
#include <fmt/base.h>

__global__ void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  cudaDeviceProp deviceProp;
  for (int i = 0; i < deviceCount; i++) {
    cudaGetDeviceProperties(&deviceProp, i);
    fmt::println("Device {}: {}", i, deviceProp.name);
  }

  int N = 1 << 20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  fmt::println("Running kernel with {} elements", N);
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3)
  int maxError = 0;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  fmt::println("Max error: {}", maxError);

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
