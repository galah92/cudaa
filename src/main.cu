#include <cuda_runtime.h>
#include <fmt/base.h>

__global__ void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

__global__ void add_block(int n, float *x, float *y) {
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

__global__ void add_grid(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

int main() {
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

  // cudaMemPrefetchAsync(x, N * sizeof(float), 0, 0);
  // cudaMemPrefetchAsync(y, N * sizeof(float), 0, 0);

  // int blockSize = 256;
  // int numBlocks = (N + blockSize - 1) / blockSize;
  // add_grid<<<numBlocks, blockSize>>>(N, x, y);
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  fmt::println("Max error: {}", maxError);

  cudaFree(x);
  cudaFree(y);
}
