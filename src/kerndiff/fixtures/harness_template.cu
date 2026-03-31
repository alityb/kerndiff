// kerndiff benchmark harness (auto-generated)

{{DTYPE_INCLUDE}}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
#ifndef BUF_ELEMS
#define BUF_ELEMS {{BUF_ELEMS}}
#endif
#ifndef GRID_SIZE
#define GRID_SIZE ((BUF_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE)
#endif

static {{ELEM_TYPE}} *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
static char *d_l2_scratch = nullptr;
static size_t d_l2_scratch_size = 0;
static const int N = BUF_ELEMS;

// Globaltimer gives nanosecond-resolution GPU wall-clock timestamps.
// Two device globals hold start/end; the two tiny kernels below read them.
// This replaces cudaEventElapsedTime for batch timing, eliminating ~1µs
// event-query overhead and giving sub-microsecond precision.
static __device__ unsigned long long _kd_ts_start;
static __device__ unsigned long long _kd_ts_end;

__global__ void _kd_read_start() {
    if (threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_kd_ts_start));
}
__global__ void _kd_read_end() {
    if (threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_kd_ts_end));
}

__global__ void _kerndiff_fill({{ELEM_TYPE}} *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = ({{ELEM_TYPE}})(i % 64 + 1);
}

// Cache-streaming flush: reads each cache line with bypass hint so L2 is
// actually evicted rather than just overwritten.
__global__ void _kerndiff_l2_flush(uint32_t *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { volatile uint32_t x = __ldcs(&buf[i]); (void)x; }
}

static void setup_buffers() {
    if (d_a) return;
    CHECK(cudaMalloc(&d_a, BUF_ELEMS * sizeof({{ELEM_TYPE}})));
    CHECK(cudaMalloc(&d_b, BUF_ELEMS * sizeof({{ELEM_TYPE}})));
    CHECK(cudaMalloc(&d_c, BUF_ELEMS * sizeof({{ELEM_TYPE}})));
    _kerndiff_fill<<<(BUF_ELEMS + 255) / 256, 256>>>(d_a, BUF_ELEMS);
    _kerndiff_fill<<<(BUF_ELEMS + 255) / 256, 256>>>(d_b, BUF_ELEMS);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemset(d_c, 0, BUF_ELEMS * sizeof({{ELEM_TYPE}})));
}

static void run_once() {
    {{KERNEL_CALL}};
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

static void flush_l2(size_t l2_size) {
    if (l2_size == 0) return;
    // Lazy allocation: allocate once, reuse on every subsequent flush call.
    if (d_l2_scratch_size < l2_size) {
        if (d_l2_scratch) CHECK(cudaFree(d_l2_scratch));
        CHECK(cudaMalloc(&d_l2_scratch, l2_size));
        d_l2_scratch_size = l2_size;
    }
    // Use cache-streaming reads to actually evict L2 lines.
    int n_words = (int)(l2_size / sizeof(uint32_t));
    _kerndiff_l2_flush<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_l2_scratch, n_words);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

static void dump_output(int count) {
    int n = count < N ? count : N;
    {{ELEM_TYPE}} *host = ({{ELEM_TYPE}} *)malloc(n * sizeof({{ELEM_TYPE}}));
    CHECK(cudaMemcpy(host, d_c, n * sizeof({{ELEM_TYPE}}), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        printf("%.6g\n", (double)host[i]);
    }
    free(host);
}

int main(int argc, char** argv) {
    int iters = 1;
    int multi_time = 0;  // --multi-time N: run N timed iterations, print N latencies
    size_t l2_flush_size = 0;
    int dump_count = 0;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--iters") == 0)
            iters = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--multi-time") == 0)
            multi_time = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--l2-flush") == 0)
            l2_flush_size = (size_t)atoll(argv[i + 1]);
        if (strcmp(argv[i], "--dump-output") == 0)
            dump_count = atoi(argv[i + 1]);
    }

    setup_buffers();

    if (dump_count > 0) {
        run_once();
        dump_output(dump_count);
        return 0;
    }

    if (multi_time > 0) {
        for (int i = 0; i < multi_time; i++) {
            if (l2_flush_size > 0)
                flush_l2(l2_flush_size);
            _kd_read_start<<<1,1>>>();
            run_once();
            _kd_read_end<<<1,1>>>();
            CHECK(cudaDeviceSynchronize());
            unsigned long long ts0, ts1;
            CHECK(cudaMemcpyFromSymbol(&ts0, _kd_ts_start, sizeof(ts0)));
            CHECK(cudaMemcpyFromSymbol(&ts1, _kd_ts_end,   sizeof(ts1)));
            printf("%.3f\n", (ts1 - ts0) / 1000.0);  // nanoseconds → microseconds
        }
    } else if (iters == 1) {
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
