#pragma once

#include "common.cuh"

// GGML_CUDA_FATTN_DEC_Q4=N: single-token decode attention straight from a q4_0 K/V cache with the position bias
// (Inkling 2) from N K/V cells on, see fattn-dec-q4.cu
bool ggml_cuda_fattn_dec_q4_applies(const ggml_tensor * dst);
void ggml_cuda_fattn_dec_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
