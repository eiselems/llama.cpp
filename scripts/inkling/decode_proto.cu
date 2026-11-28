// Prototype: one Inkling 2 decode token (48 layer applications) driven two ways, with timings from the S32 nsys profile.
//  host-driven (as llama.cpp today): per layer sync -> plan -> [uploads] -> launch graph A -> CPU experts -> copy result
//                                    -> launch graph C (router of the next layer) -> D2H ids
//  gpu-driven: all work of a layer is queued ahead; the GPU waits on flags in pinned host memory (cuStreamWaitValue32),
//              the host only spins on the router ids, plans, computes the CPU experts and sets flags.
// Usage: decode_proto [tokens]
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)
#define CD(x) do { CUresult err_ = (x); if (err_ != CUDA_SUCCESS) { const char * es_; cuGetErrorString(err_, &es_); printf("%s:%d %s\n", __FILE__, __LINE__, es_); exit(1); } } while (0)

static const int    L = 48;
static const double A_US = 95, C_US = 210, TAIL_US = 1300;  // graph A, graph C, LM head + sampling per token
static const int    A_NODES = 10, C_NODES = 28;
static const size_t EXPERT = 4478976, RESULT = 2304 * 4 * 8, IDS = 64;

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void spin(long long ns) {  // wall-clock spin, independent of the SM clock
    unsigned long long t = gtimer();
    while ((long long) (gtimer() - t) < ns) {}
}

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void busy(double us) { double t = now_us(); while (now_us() - t < us) {} }

static double cyc_per_us;
static cudaGraphExec_t capture(cudaStream_t s, int nodes, double ns_each) {
    cudaGraph_t g; cudaGraphExec_t e;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
    for (int i = 0; i < nodes; i++) spin<<<1, 32, 0, s>>>((long long) ns_each);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&e, g, 0));
    return e;
}
// a graph of `nodes` kernels that takes `us` on the GPU including the gaps between its kernels (as nsys measured them)
static cudaGraphExec_t make_graph(cudaStream_t s, int nodes, double us) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double each = us * 1000.0 / nodes;
    cudaGraphExec_t e = nullptr;
    for (int it = 0; it < 6; it++) {
        if (e) CK(cudaGraphExecDestroy(e));
        e = capture(s, nodes, each);
        CK(cudaGraphLaunch(e, s)); CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(a, s));
        for (int r = 0; r < 20; r++) CK(cudaGraphLaunch(e, s));
        CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const double got = ms * 1000.0 / 20;
        each = std::max(0.0, each + (us - got) * 1000.0 / nodes);
    }
    return e;
}

int main(int argc, char ** argv) {
    const int tokens = argc > 1 ? atoi(argv[1]) : 60;
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CK(cudaFree(0));
    cudaStream_t s, up;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking));

    // calibrate the spin kernel
    {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        spin<<<1, 32, 0, s>>>(1000000); CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(a, s)); spin<<<1, 32, 0, s>>>(10000000); CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); cyc_per_us = 10000000.0 / (ms * 1000.0);
    }
    cudaGraphExec_t gA = make_graph(s, A_NODES, A_US), gC = make_graph(s, C_NODES, C_US), gT = make_graph(s, 20, TAIL_US);

    // per layer: uploads and CPU expert time, the S32 decode pattern (28 layers no upload, 15 one, 5 two or more)
    int nup[L]; double cpu_us[L];
    for (int i = 0; i < L; i++) {
        const int k = (i * 7) % L;
        nup[i] = k < 28 ? 0 : k < 43 ? 1 : 3;
        cpu_us[i] = nup[i] == 0 ? 90 : nup[i] == 1 ? 215 : 490;
        if (getenv("NO_UP"))  nup[i] = 0;
        if (getenv("NO_CPU")) cpu_us[i] = 0;
    }

    void *d_up, *h_up, *d_res, *h_res, *d_ids, *h_ids, *d_map, *h_map;
    CK(cudaMalloc(&d_up, 3 * EXPERT)); CK(cudaMallocHost(&h_up, 3 * EXPERT));
    CK(cudaMalloc(&d_res, RESULT));    CK(cudaMallocHost(&h_res, RESULT));
    CK(cudaMalloc(&d_ids, IDS));       CK(cudaMallocHost(&h_ids, IDS));
    CK(cudaMalloc(&d_map, 512));       CK(cudaMallocHost(&h_map, 512));
    memset(h_up, 1, 3 * EXPERT); memset(h_res, 0, RESULT);

    // flags in mapped pinned memory: [0] ids ready, [1] plan done, [2] cpu result ready
    volatile uint32_t * flags; CK(cudaHostAlloc((void **) &flags, 64, cudaHostAllocMapped));
    memset((void *) flags, 0, 64);
    CUdeviceptr f_dev; CD(cuMemHostGetDevicePointer(&f_dev, (void *) flags, 0));
    const CUdeviceptr f_ids = f_dev, f_plan = f_dev + 4, f_cpu = f_dev + 8;

    // ---------------- host-driven ----------------
    bool lean = false; // no small copies: kernels read and write pinned memory directly (zero copy)
    double t_sync = 0, t_cpu = 0, t_api = 0; int n_tok = 0;
    auto host_token = [&]() {
        n_tok++;
        for (int l = 0; l < L; l++) {
            double t0 = now_us();
            CK(cudaStreamSynchronize(s));                       // router ids of this layer (D2H queued before)
            t_sync += now_us() - t0;
            busy(8);                                            // plan the cache
            if (nup[l]) CK(cudaMemcpyAsync(d_up, h_up, nup[l] * EXPERT, cudaMemcpyHostToDevice, s));
            if (!lean) CK(cudaMemcpyAsync(d_map, h_map, 512, cudaMemcpyHostToDevice, s));
            double t1 = now_us();
            CK(cudaGraphLaunch(gA, s));
            double t2 = now_us();
            busy(cpu_us[l]);                                    // CPU experts
            double t3 = now_us();
            if (!lean) CK(cudaMemcpyAsync(d_res, h_res, RESULT, cudaMemcpyHostToDevice, s));
            CK(cudaGraphLaunch(gC, s));
            if (!lean) CK(cudaMemcpyAsync(h_ids, d_ids, IDS, cudaMemcpyDeviceToHost, s));
            t_api += (t2 - t1) + (now_us() - t3); t_cpu += t3 - t2;
        }
        CK(cudaGraphLaunch(gT, s));
        CK(cudaStreamSynchronize(s));                           // the sampled token
    };

    // ---------------- gpu-driven ----------------
    uint32_t seq = 0;
    auto enqueue_layer = [&](uint32_t q) {                      // static work of one layer, gated by flags
        CD(cuStreamWaitValue32((CUstream) s, f_plan, q, CU_STREAM_WAIT_VALUE_GEQ));
        if (!lean) CK(cudaMemcpyAsync(d_map, h_map, 512, cudaMemcpyHostToDevice, s)); // reads the pinned map when it runs
        CK(cudaGraphLaunch(gA, s));
        CD(cuStreamWaitValue32((CUstream) s, f_cpu, q, CU_STREAM_WAIT_VALUE_GEQ));
        if (!lean) CK(cudaMemcpyAsync(d_res, h_res, RESULT, cudaMemcpyHostToDevice, s));
        CK(cudaGraphLaunch(gC, s));
        if (!lean) CK(cudaMemcpyAsync(h_ids, d_ids, IDS, cudaMemcpyDeviceToHost, s));
        CD(cuStreamWriteValue32((CUstream) s, f_ids, q + 1, CU_STREAM_WRITE_VALUE_DEFAULT));
    };
    auto set_flag = [&](int i, uint32_t v) { std::atomic_thread_fence(std::memory_order_seq_cst); flags[i] = v; };
    auto gpu_token = [&]() {
        // ids of layer 0 come from the previous token: mark them ready, queue layer 0
        set_flag(0, seq + 1);
        enqueue_layer(seq + 1);
        for (int l = 0; l < L; l++) {
            const uint32_t q = seq + 1 + l;
            while (flags[0] < q) {}                             // spin on the router ids of this layer
            busy(8);                                            // plan the cache
            if (nup[l]) {
                CK(cudaMemcpyAsync(d_up, h_up, nup[l] * EXPERT, cudaMemcpyHostToDevice, up));
                CD(cuStreamWriteValue32((CUstream) up, f_plan, q, CU_STREAM_WRITE_VALUE_DEFAULT));
            } else {
                set_flag(1, q);
            }
            if (l + 1 < L) enqueue_layer(q + 1);                // next layer, off the critical path
            busy(cpu_us[l]);                                    // CPU experts
            set_flag(2, q);
        }
        seq += L;
        CK(cudaGraphLaunch(gT, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaStreamSynchronize(up));
    };

    double gpu_busy = L * (A_US + C_US) + TAIL_US;
    double up_total = 0; for (int i = 0; i < L; i++) up_total += nup[i] * EXPERT / 26.3e3;
    printf("per token: GPU kernels %.2f ms, uploads %.2f ms, CPU experts %.2f ms (each alone)\n", gpu_busy / 1e3, up_total / 1e3,
           [&] { double c = 0; for (int i = 0; i < L; i++) c += cpu_us[i]; return c / 1e3; }());

    for (int round = 0; round < 3; round++) {
        for (int mode = 0; mode < 4; mode++) {
            lean = mode >= 2;
            for (int t = 0; t < 5; t++) (mode & 1) ? gpu_token() : host_token();   // warm up
            double t0 = now_us();
            for (int t = 0; t < tokens; t++) (mode & 1) ? gpu_token() : host_token();
            double ms = (now_us() - t0) / tokens / 1e3;
            printf("round %d %-12s %.3f ms/token = %.1f t/s\n", round, (const char *[]){"host-driven", "gpu-driven", "host-lean", "gpu-lean"}[mode], ms, 1e3 / ms);
            if (!(mode & 1)) { printf("   host-driven per token: sync %.2f ms, cpu busy %.2f ms, launch api %.2f ms\n", t_sync/n_tok/1e3, t_cpu/n_tok/1e3, t_api/n_tok/1e3); t_sync = t_cpu = t_api = 0; n_tok = 0; }
        }
    }
    return 0;
}
