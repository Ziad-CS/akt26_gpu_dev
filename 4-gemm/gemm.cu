#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error: %s\n", \
                     cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error: %d\n", \
                     static_cast<int>(status)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Custom batched GEMM: C[b] = A[b] * B[b]
// Matrices are column-major.
__global__ void batch_gemm_kernel(
    int n, int batch_size,
    const float* A, const float* B, float* C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    if (row >= n || col >= n || b >= batch_size)
        return;

    size_t base = (size_t)b * n * n;
    float sum = 0.0f;

    for (int k = 0; k < n; ++k) {
        sum += A[base + (size_t)k * n + row] *
               B[base + (size_t)col * n + k];
    }

    C[base + (size_t)col * n + row] = sum;
}

void batched_gemm_cpu(
    int n, int batch_size,
    const float* A, const float* B, float* C)
{
    size_t matrix_size = (size_t)n * n;

    for (int b = 0; b < batch_size; ++b) {
        size_t base = (size_t)b * matrix_size;

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                float sum = 0.0f;

                for (int k = 0; k < n; ++k) {
                    sum += A[base + (size_t)k * n + row] *
                           B[base + (size_t)col * n + k];
                }

                C[base + (size_t)col * n + row] = sum;
            }
        }
    }
}

double forward_error(
    int n, const float* result, const float* reference)
{
    double diff2 = 0.0, ref2 = 0.0;
    size_t size = (size_t)n * n;

    for (size_t i = 0; i < size; ++i) {
        double diff = (double)result[i] - reference[i];
        diff2 += diff * diff;
        ref2 += (double)reference[i] * reference[i];
    }

    double diff_norm = std::sqrt(diff2);
    double ref_norm = std::sqrt(ref2);

    if (ref_norm > 0.0)
        return diff_norm / (n * ref_norm);

    return diff_norm == 0.0 ? 0.0 : INFINITY;
}

double max_batch_forward_error(
    int n, int batch_size,
    const float* result, const float* reference)
{
    size_t matrix_size = (size_t)n * n;
    double max_error = 0.0;

    for (int b = 0; b < batch_size; ++b) {
        size_t base = (size_t)b * matrix_size;
        double e = forward_error(
            n, result + base, reference + base);
        max_error = std::max(max_error, e);
    }

    return max_error;
}

int main(int argc, char** argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int repeats = 100;

    if (n <= 0 || n > 32 || batch_size <= 0 ||
        batch_size > 65535) {
        std::fprintf(stderr,
            "Usage: %s matrix_size(1..32) batch_size(1..65535)\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t bytes = total * sizeof(float);

    std::vector<float> h_A(total), h_B(total), h_ref(total);
    std::vector<float> h_custom(total), h_cublas(total);

    for (size_t i = 0; i < total; ++i) {
        h_A[i] = std::sin(0.013f * (float)(i + 1));
        h_B[i] = std::cos(0.017f * (float)(i + 1));
    }

    batched_gemm_cpu(n, batch_size,
                     h_A.data(), h_B.data(), h_ref.data());

    float *d_A = nullptr, *d_B = nullptr;
    float *d_custom = nullptr, *d_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_custom, bytes));
    CUDA_CHECK(cudaMalloc(&d_cublas, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes,
                          cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("GPU: %s (compute capability %d.%d)\n",
                prop.name, prop.major, prop.minor);
    std::printf("Matrix size: %d x %d, batch size: %d\n",
                n, n, batch_size);
    std::printf("FP32, column-major matrices, repeats: %d\n\n",
                repeats);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y,
              batch_size);

    batch_gemm_kernel<<<grid, block>>>(
        n, batch_size, d_A, d_B, d_custom);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        batch_gemm_kernel<<<grid, block>>>(
            n, batch_size, d_A, d_B, d_custom);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));
    double custom_us = custom_ms * 1000.0 / repeats;

    CUDA_CHECK(cudaMemcpy(h_custom.data(), d_custom, bytes,
                          cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    const long long stride = (long long)matrix_size;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n, &alpha,
        d_A, n, stride,
        d_B, n, stride,
        &beta, d_cublas, n, stride,
        batch_size));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n, &alpha,
            d_A, n, stride,
            d_B, n, stride,
            &beta, d_cublas, n, stride,
            batch_size));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    double cublas_us = cublas_ms * 1000.0 / repeats;

    CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_cublas, bytes,
                          cudaMemcpyDeviceToHost));

    double flops = 2.0 * n * n * n * batch_size;
    double custom_gflops = flops / (custom_us * 1.0e3);
    double cublas_gflops = flops / (cublas_us * 1.0e3);

    double custom_error = max_batch_forward_error(
        n, batch_size, h_custom.data(), h_ref.data());
    double cublas_error = max_batch_forward_error(
        n, batch_size, h_cublas.data(), h_ref.data());

    std::printf("%-24s %12s %12s %20s\n",
                "Implementation", "Time (us)", "GFLOP/s",
                "Max batch fwd error");
    std::printf("-----------------------------------------------------------------------\n");
    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "Custom kernel", custom_us,
                custom_gflops, custom_error);
    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "cuBLAS Strided Batched", cublas_us,
                cublas_gflops, cublas_error);

    const double tol = 1e-6;
    if (custom_error <= tol && cublas_error <= tol) {
        std::printf("\nTEST PASS\n");
        std::printf("Speedup (cuBLAS time / custom time): %.3fx\n",
                    cublas_us / custom_us);
    } else {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_custom));
    CUDA_CHECK(cudaFree(d_cublas));

    return 0;
}
