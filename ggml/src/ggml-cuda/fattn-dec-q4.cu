// Single-token decode attention straight from a q4_0 K/V cache, for the relative-position bias of Inkling 2
// (GGML_CUDA_FATTN_DEC_Q4=N, from N K/V cells on).
//
// The general path converts all of K and V to F16 for every token and then runs the tile kernel on the copies, so
// a 24k context moves ~4x the q4_0 bytes per layer. Here one CUDA block takes a range of cells of one K/V head for
// all the Q heads of that K/V head (GQA), dequantizes each K and V row once in registers and keeps the softmax state
// in F32. A second kernel merges the ranges in a fixed order (deterministic).
//
// Math per Q head h: l_c = scale*dot(q_h, k_c) + half(mask_c + table[bin_c, h]), softmax over the cells and the sink,
// out = sum_c p_c v_c. The masked cells (-inf) are skipped. Same formula as the tile kernel, in F32 instead of F16,
// so the results differ from it in the last bits.

#include "common.cuh"
#include "fattn-dec-q4.cuh"

#include <cstdlib>

#define DEC_Q4_D        128                  // head size
#define DEC_Q4_LANES    8                    // lanes per cell, 16 dims each
#define DEC_Q4_NWARPS   4
#define DEC_Q4_GROUPS   (DEC_Q4_NWARPS*WARP_SIZE/DEC_Q4_LANES) // cells in flight per block
#define DEC_Q4_MAX_GQA  4

// 16 dims of a q4_0 row: lane l of the cell takes q4_0 block l/2, bytes 8*(l%2) .. +7 -> elements j and j + 16
static __device__ __forceinline__ void dec_q4_load(const block_q4_0 * row, const int l, float v[16]) {
    const block_q4_0 * b = row + l/2;
    const float d = __half2float(b->d);
    const int j0 = 8*(l % 2);
    // 8 bytes, 2-byte aligned only (block_q4_0 is 18 bytes): 4 16-bit loads
    const uint16_t * q16 = (const uint16_t *) (b->qs + j0);
    uint16_t q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = q16[i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int byte = (q[i/2] >> (8*(i % 2))) & 0xFF;
        v[i]     = d*float((byte & 0x0F) - 8);
        v[i + 8] = d*float((byte >>   4) - 8);
    }
}

template <int gqa>
__launch_bounds__(DEC_Q4_NWARPS*WARP_SIZE, 1)
static __global__ void fattn_dec_q4_partial(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        const float * __restrict__ table, const int32_t * __restrict__ positions,
        float * __restrict__ part_acc, float2 * __restrict__ part_meta,
        const int n_kv, const int cells_per_block, const float scale,
        const int64_t q_nb2, const int64_t k_nb1, const int64_t k_nb2, const int64_t v_nb1, const int64_t v_nb2,
        const int64_t t_s_head, const int64_t t_s_bin, const int n_bins, const int n_pos_cols) {
    const int split = blockIdx.x;
    const int hkv   = blockIdx.y;
    const int tid   = threadIdx.x;
    const int group = tid / DEC_Q4_LANES;
    const int l     = tid % DEC_Q4_LANES;
    // the cell groups of a warp diverge (different cells, masked cells): shuffle within the group only
    const unsigned group_mask = ((1u << DEC_Q4_LANES) - 1) << (DEC_Q4_LANES*(group % (WARP_SIZE/DEC_Q4_LANES)));

    // this lane's 16 dims of each Q head: element j0 + i and j0 + 16 + i of q4_0 block l/2, see dec_q4_load
    const int dim0 = 32*(l/2) + 8*(l % 2);
    float q[gqa][16];
#pragma unroll
    for (int g = 0; g < gqa; ++g) {
        const float * qh = (const float *) ((const char *) Q + (hkv*gqa + g)*q_nb2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[g][i]     = qh[dim0 + i]*scale;
            q[g][i + 8] = qh[dim0 + 16 + i]*scale;
        }
    }

    const int32_t pq = positions[n_pos_cols]; // the query position (row 1, column 0)

    float M[gqa], S[gqa], acc[gqa][16];
#pragma unroll
    for (int g = 0; g < gqa; ++g) {
        M[g] = -INFINITY;
        S[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[g][i] = 0.0f;
        }
    }

    const int c_begin = split*cells_per_block;
    const int c_end   = min(n_kv, c_begin + cells_per_block);
    for (int c = c_begin + group; c < c_end; c += DEC_Q4_GROUPS) {
        const float m = __half2float(mask[c]);
        if (isinf(m) && m < 0.0f) {
            continue; // the whole group of 8 lanes takes the same branch
        }
        float k[16];
        dec_q4_load((const block_q4_0 *) (K + c*k_nb1 + hkv*k_nb2), l, k);
        float logit[gqa];
#pragma unroll
        for (int g = 0; g < gqa; ++g) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                s = fmaf(q[g][i], k[i], s);
            }
#pragma unroll
            for (int o = DEC_Q4_LANES/2; o > 0; o >>= 1) {
                s += __shfl_xor_sync(group_mask, s, o, DEC_Q4_LANES);
            }
            logit[g] = s;
        }
        // bias bin from the positions, as fattn_bias_bin
        const int32_t pc = positions[c];
        const int64_t dpos = (int64_t) pq - pc;
        const int extent = n_bins - 1;
        const int bin = pc >= 0 && dpos >= 0 && dpos < extent ? (int) dpos : extent;
        float v[16];
        dec_q4_load((const block_q4_0 *) (V + c*v_nb1 + hkv*v_nb2), l, v);
#pragma unroll
        for (int g = 0; g < gqa; ++g) {
            const float mb = __half2float(__float2half(m + table[bin*t_s_bin + (hkv*gqa + g)*t_s_head]));
            const float x  = logit[g] + mb;
            const float Mn = fmaxf(M[g], x);
            const float a  = expf(M[g] - Mn); // M = -inf at the start: a = 0
            const float p  = expf(x - Mn);
            S[g] = S[g]*a + p;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[g][i] = fmaf(p, v[i], acc[g][i]*a);
            }
            M[g] = Mn;
        }
    }

    // merge the groups of the block in a fixed order
    __shared__ float sM[DEC_Q4_GROUPS][gqa];
    __shared__ float sS[DEC_Q4_GROUPS][gqa];
    __shared__ float sAcc[DEC_Q4_GROUPS][gqa][DEC_Q4_D];
    if (l == 0) {
#pragma unroll
        for (int g = 0; g < gqa; ++g) {
            sM[group][g] = M[g];
            sS[group][g] = S[g];
        }
    }
#pragma unroll
    for (int g = 0; g < gqa; ++g) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sAcc[group][g][dim0 + i]      = acc[g][i];
            sAcc[group][g][dim0 + 16 + i] = acc[g][i + 8];
        }
    }
    __syncthreads();

    for (int e = tid; e < gqa*DEC_Q4_D; e += blockDim.x) {
        const int g = e / DEC_Q4_D;
        const int d = e % DEC_Q4_D;
        float Mb = -INFINITY;
        for (int r = 0; r < DEC_Q4_GROUPS; ++r) {
            Mb = fmaxf(Mb, sM[r][g]);
        }
        float Sb = 0.0f, Ab = 0.0f;
        if (Mb != -INFINITY) {
            for (int r = 0; r < DEC_Q4_GROUPS; ++r) {
                if (sS[r][g] > 0.0f) {
                    const float a = expf(sM[r][g] - Mb);
                    Sb = fmaf(sS[r][g], a, Sb);
                    Ab = fmaf(sAcc[r][g][d], a, Ab);
                }
            }
        }
        const int h = hkv*gqa + g;
        part_acc[((int64_t) h*gridDim.x + split)*DEC_Q4_D + d] = Ab;
        if (d == 0) {
            part_meta[(int64_t) h*gridDim.x + split] = make_float2(Mb, Sb);
        }
    }
}

// one block per Q head: merge the splits in order, add the sink, normalize
static __global__ void fattn_dec_q4_combine(
        const float * __restrict__ part_acc, const float2 * __restrict__ part_meta, const float * __restrict__ sinks,
        float * __restrict__ dst, const int n_split) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        const float2 ms = part_meta[(int64_t) h*n_split + s];
        if (ms.y > 0.0f) {
            M = fmaxf(M, ms.x);
        }
    }
    float S = sinks ? expf(sinks[h] - M) : 0.0f;
    float A = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float2 ms = part_meta[(int64_t) h*n_split + s];
        if (ms.y > 0.0f) {
            const float a = expf(ms.x - M);
            S = fmaf(ms.y, a, S);
            A = fmaf(part_acc[((int64_t) h*n_split + s)*DEC_Q4_D + d], a, A);
        }
    }
    dst[(int64_t) h*DEC_Q4_D + d] = S > 0.0f ? A/S : 0.0f;
}

bool ggml_cuda_fattn_dec_q4_applies(const ggml_tensor * dst) {
    static const int64_t min_kv = [] { const char * e = getenv("GGML_CUDA_FATTN_DEC_Q4"); return e ? (int64_t) atoll(e) : (int64_t) 0; }();
    if (min_kv <= 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * table = dst->src[6];
    const ggml_tensor * pos = dst->src[7];
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (Q->ne[0] != DEC_Q4_D || Q->ne[1] != 1 || Q->ne[3] != 1 || Q->type != GGML_TYPE_F32 ||
            K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || K->ne[0] != DEC_Q4_D || V->ne[0] != DEC_Q4_D ||
            K->ne[1] < min_kv || K->ne[3] != 1 || V->ne[3] != 1 || V->ne[1] != K->ne[1] ||
            Q->ne[2] % K->ne[2] != 0 || Q->ne[2]/K->ne[2] > DEC_Q4_MAX_GQA || K->ne[2] != V->ne[2] ||
            !mask || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
            dst->src[5] || !table || !pos || !ggml_flash_attn_ext_bias_from_positions(dst) ||
            pos->ne[0] != K->ne[1] || table->ne[2] != 1 ||
            max_bias != 0.0f || softcap != 0.0f || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            K->nb[0] != sizeof(block_q4_0) || V->nb[0] != sizeof(block_q4_0) ||
            (dst->src[4] && dst->src[4]->type != GGML_TYPE_F32)) {
        return false;
    }
    return true;
}

void ggml_cuda_fattn_dec_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * table = dst->src[6];
    const ggml_tensor * pos = dst->src[7];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_kv   = (int) K->ne[1];
    const int n_hkv  = (int) K->ne[2];
    const int gqa    = (int) (Q->ne[2] / n_hkv);
    const int target = 4*ggml_cuda_info().devices[ctx.device].nsm; // blocks
    int n_split = std::max(1, std::min((target + n_hkv - 1)/n_hkv, (n_kv + 63)/64));
    const int cells_per_block = (n_kv + n_split - 1)/n_split;
    n_split = (n_kv + cells_per_block - 1)/cells_per_block;

    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool(), (size_t) Q->ne[2]*n_split*DEC_Q4_D);
    ggml_cuda_pool_alloc<float2> part_meta(ctx.pool(), (size_t) Q->ne[2]*n_split);

    cudaStream_t stream = ctx.stream();
    const dim3 grid(n_split, n_hkv, 1);
    const dim3 block(DEC_Q4_NWARPS*WARP_SIZE, 1, 1);
#define DEC_Q4_LAUNCH(G) fattn_dec_q4_partial<G><<<grid, block, 0, stream>>>( \
        (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const half *) mask->data, \
        (const float *) table->data, (const int32_t *) pos->data, part_acc.ptr, part_meta.ptr, \
        n_kv, cells_per_block, scale, Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], \
        table->nb[0]/(int64_t) sizeof(float), table->nb[1]/(int64_t) sizeof(float), (int) table->ne[1], (int) pos->ne[0])
    switch (gqa) {
        case 1: DEC_Q4_LAUNCH(1); break;
        case 2: DEC_Q4_LAUNCH(2); break;
        case 3: DEC_Q4_LAUNCH(3); break;
        case 4: DEC_Q4_LAUNCH(4); break;
        default: GGML_ABORT("fatal error");
    }
#undef DEC_Q4_LAUNCH
    CUDA_CHECK(cudaGetLastError());

    fattn_dec_q4_combine<<<(int) Q->ne[2], DEC_Q4_D, 0, stream>>>(part_acc.ptr, part_meta.ptr,
        sinks ? (const float *) sinks->data : nullptr, (float *) dst->data, n_split);
    CUDA_CHECK(cudaGetLastError());
}
