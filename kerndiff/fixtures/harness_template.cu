// kerndiff benchmark harness
// {{KERNEL_SOURCE}} -> full .cu file contents
// {{KERNEL_NAME}}   -> kernel function name (used in default stub only)
// {{KERNEL_CALL}}   -> full launch expression, e.g. my_kernel<<<G,B>>>(d_a, d_b, d_c, N)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

{{KERNEL_SOURCE}}

#define CHECK(call)                                                       \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
              __FILE__, __LINE__, cudaGetErrorString(_e));                \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef GRID_SIZE
#define GRID_SIZE 1024
#endif
#ifndef BUF_ELEMS
#define BUF_ELEMS (1 << 22)
#endif

static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
static const int N = BUF_ELEMS;

static void setup_buffers() {
    if (d_a) return;
    CHECK(cudaMalloc(&d_a, BUF_ELEMS * sizeof(float)));
    CHECK(cudaMalloc(&d_b, BUF_ELEMS * sizeof(float)));
    CHECK(cudaMalloc(&d_c, BUF_ELEMS * sizeof(float)));
    CHECK(cudaMemset(d_a, 0, BUF_ELEMS * sizeof(float)));
    CHECK(cudaMemset(d_b, 0, BUF_ELEMS * sizeof(float)));
}

static void run_once() {
    {{KERNEL_CALL}};
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

static void flush_l2(size_t l2_size) {
    char *scratch;
    CHECK(cudaMalloc(&scratch, l2_size));
    CHECK(cudaMemset(scratch, 0, l2_size));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(scratch));
}

int main(int argc, char** argv) {
    int iters = 1;
    size_t l2_flush_size = 0;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--iters") == 0)
            iters = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--l2-flush") == 0)
            l2_flush_size = (size_t)atoll(argv[i + 1]);
    }

    setup_buffers();

    if (iters == 1) {
        if (l2_flush_size > 0)
            flush_l2(l2_flush_size);
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        run_once();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        printf("%.3f\n", ms * 1000.0f);  // microseconds to stdout
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));
    } else {
        for (int i = 0; i < iters; i++) run_once();
    }
    return 0;
}
