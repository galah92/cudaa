#include <chrono>
#include <cuda_runtime.h>
#include <fmt/base.h>

__global__ void sgemm(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float value = 0.0f;
    for (int k = 0; k < K; k++) {
      value += A[x * K + k] * B[k * N + y];
    }
    C[x * N + y] = alpha * value + beta * C[x * N + y];
  }
}

template <const int BLOCKSIZE>
__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
  int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  int y = blockIdx.y * blockDim.y + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N) {
    float value = 0.0f;
    for (int k = 0; k < K; k++) {
      value += A[x * K + k] * B[k * N + y];
    }
    C[x * N + y] = alpha * value + beta * C[x * N + y];
  }
}

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

using kernel_t = decltype(&sgemm);
kernel_t kernels[] = {
    sgemm,
    sgemm_coalesce<32>,
    sgemm_shared_mem<32>,
};

void rand_matrix(int M, int N, float *mat) {
  for (int i = 0; i < M * N; i++) {
    mat[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

int main() {
  const auto kernel_idx = 2;
  fmt::println("Using kernel {}", kernel_idx);

  const auto SIZES = {128, 256, 512, 1024, 2048};

  for (const int size : SIZES) {
    const int M = size, N = size, K = size;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    float *A, *B, *C;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    // initialize A, B and C arrays on the host
    rand_matrix(M, K, A);
    rand_matrix(K, N, B);
    rand_matrix(M, N, C);

    cudaMemPrefetchAsync(A, M * K * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(B, K * N * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(C, M * N * sizeof(float), 0, 0);

    const auto blockSize = 16;
    const dim3 blockDim(blockSize, blockSize);
    const dim3 gridDim(ceil_div(M, blockSize), ceil_div(N, blockSize));
    const auto kernel = kernels[kernel_idx];

    const auto repeats = 50;
    const auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeats; i++) {
      kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host

    const auto end_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float> duration = end_time - start_time;
    const auto time = duration.count() / repeats;
    fmt::println("Time for SGEMM of size {:4}x{:4}: {:.5f}s", size, size, time);

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }
}
