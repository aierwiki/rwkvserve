/*
 * RWKV7 Infinite Context CUDA Kernel - Recompute Mode Only
 * 
 * 只保留重计算（Recompute）模式，用于最大程度节省显存
 * 
 * 状态维度: [V, K] 矩阵
 *   - 每个线程 v 维护 state[v, :]，即状态矩阵的第 v 行（K 个元素）
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "ATen/ATen.h"

using bf = __nv_bfloat16;
using fp16 = __half;

__device__ inline float safe_exp(float x) {
    x = fmaxf(fminf(x, 80.0f), -80.0f);
    return __expf(x);
}

/*
 * Forward Kernel
 * 
 * 状态在输入/输出时共用同一块内存（in-place 更新）
 */
template <typename F>
__global__ void forward_kernel_infctx(
    int B, int T, int H, int C,
    const F* __restrict__ w_,
    const F* __restrict__ q_,
    const F* __restrict__ k_,
    const F* __restrict__ v_,
    const F* __restrict__ a_,
    const F* __restrict__ b_,
    float* __restrict__ s_,
    F* __restrict__ y_
) {
    int bb = blockIdx.y;
    int hh = blockIdx.x;
    int v_idx = threadIdx.x;
    
    if (bb >= B || hh >= H || v_idx >= C) return;
    
    float state[256];
    
    int state_base = ((bb * H + hh) * C + v_idx) * C;
    for (int k = 0; k < C; k++) {
        state[k] = s_[state_base + k];
    }
    
    __shared__ float q_s[256], k_s[256], w_s[256], a_s[256], b_s[256];
    
    for (int t = 0; t < T; t++) {
        int ind = ((bb * T + t) * H + hh) * C;
        
        __syncthreads();
        q_s[v_idx] = float(q_[ind + v_idx]);
        w_s[v_idx] = safe_exp(-safe_exp(float(w_[ind + v_idx])));
        k_s[v_idx] = float(k_[ind + v_idx]);
        a_s[v_idx] = float(a_[ind + v_idx]);
        b_s[v_idx] = float(b_[ind + v_idx]);
        __syncthreads();
        
        float v_val = float(v_[ind + v_idx]);
        
        // sa = state @ a
        float sa = 0.0f;
        for (int k = 0; k < C; k++) {
            sa += state[k] * a_s[k];
        }
        
        // state = w * state + sa * b + k * v
        for (int k = 0; k < C; k++) {
            state[k] = w_s[k] * state[k] + sa * b_s[k] + k_s[k] * v_val;
        }
        
        // y = state @ q
        float y_val = 0.0f;
        for (int k = 0; k < C; k++) {
            y_val += state[k] * q_s[k];
        }
        
        y_[ind + v_idx] = F(y_val);
    }
    
    // 写回最终状态
    for (int k = 0; k < C; k++) {
        s_[state_base + k] = state[k];
    }
}

/*
 * Backward Kernel with Recomputation
 * 
 * 使用分段（segment）策略减少重计算开销：
 * - 将序列分成多个 segment（默认 16 个 token）
 * - 每个 segment 只需从初始状态重计算到 segment 起点
 * - 然后正向计算 segment 内的所有状态
 * - 最后反向遍历计算梯度
 */
template <typename F>
__global__ void backward_kernel_recompute(
    int B, int T, int H, int C,
    const F* __restrict__ w_,
    const F* __restrict__ q_,
    const F* __restrict__ k_,
    const F* __restrict__ v_,
    const F* __restrict__ a_,
    const F* __restrict__ b_,
    const F* __restrict__ dy_,
    const float* __restrict__ s0_,
    const float* __restrict__ ds_in_,
    F* __restrict__ dw_,
    F* __restrict__ dq_,
    F* __restrict__ dk_,
    F* __restrict__ dv_,
    F* __restrict__ da_,
    F* __restrict__ db_,
    float* __restrict__ ds_out_
) {
    const int SEG_LEN = 16;
    
    int bb = blockIdx.y;
    int hh = blockIdx.x;
    int v_idx = threadIdx.x;
    
    if (bb >= B || hh >= H || v_idx >= C) return;
    
    // 初始化 dstate（来自后续 chunk 的梯度）
    float dstate[256];
    int state_base = ((bb * H + hh) * C + v_idx) * C;
    for (int k = 0; k < C; k++) {
        dstate[k] = (ds_in_ != nullptr) ? ds_in_[state_base + k] : 0.0f;
    }
    
    int num_segs = (T + SEG_LEN - 1) / SEG_LEN;
    
    __shared__ float q_s[256], k_s[256], w_s[256], a_s[256], b_s[256];
    __shared__ float dy_s[256], v_s[256], w_raw_s[256];
    __shared__ float sa_s[256], dsa_s[256];
    
    // Reduction buffer
    extern __shared__ float dq_buf[];
    
    // 每个 segment 的状态序列（最多 17 个状态：0 到 SEG_LEN）
    float seg_states[17][256];
    
    // 从最后一个 segment 开始反向处理
    for (int seg = num_segs - 1; seg >= 0; seg--) {
        int seg_start = seg * SEG_LEN;
        int seg_end = min(seg_start + SEG_LEN, T);
        int seg_len = seg_end - seg_start;
        
        // === 重计算 segment 开始的状态 ===
        float state[256];
        for (int k = 0; k < C; k++) {
            state[k] = s0_[state_base + k];
        }
        
        // 从初始状态正向计算到 segment 起点
        for (int t = 0; t < seg_start; t++) {
            int ind = ((bb * T + t) * H + hh) * C;
            
            __syncthreads();
            w_s[v_idx] = safe_exp(-safe_exp(float(w_[ind + v_idx])));
            k_s[v_idx] = float(k_[ind + v_idx]);
            a_s[v_idx] = float(a_[ind + v_idx]);
            b_s[v_idx] = float(b_[ind + v_idx]);
            __syncthreads();
            
            float v_val = float(v_[ind + v_idx]);
            
            float sa = 0.0f;
            for (int k = 0; k < C; k++) {
                sa += state[k] * a_s[k];
            }
            
            for (int k = 0; k < C; k++) {
                state[k] = w_s[k] * state[k] + sa * b_s[k] + k_s[k] * v_val;
            }
        }
        
        // 保存 segment 起始状态
        for (int k = 0; k < C; k++) {
            seg_states[0][k] = state[k];
        }
        
        // === 正向计算 segment 内的所有状态 ===
        for (int st = 0; st < seg_len; st++) {
            int t = seg_start + st;
            int ind = ((bb * T + t) * H + hh) * C;
        
            __syncthreads();
            w_s[v_idx] = safe_exp(-safe_exp(float(w_[ind + v_idx])));
            k_s[v_idx] = float(k_[ind + v_idx]);
            a_s[v_idx] = float(a_[ind + v_idx]);
            b_s[v_idx] = float(b_[ind + v_idx]);
            __syncthreads();
        
            float v_val = float(v_[ind + v_idx]);
            
            float sa = 0.0f;
            for (int k = 0; k < C; k++) {
                sa += state[k] * a_s[k];
            }
            
            for (int k = 0; k < C; k++) {
                state[k] = w_s[k] * state[k] + sa * b_s[k] + k_s[k] * v_val;
            }
            
            // 保存当前状态
            for (int k = 0; k < C; k++) {
                seg_states[st + 1][k] = state[k];
            }
        }
        
        // === 反向遍历 segment 计算梯度 ===
        for (int st = seg_len - 1; st >= 0; st--) {
            int t = seg_start + st;
            int ind = ((bb * T + t) * H + hh) * C;
            
            // 加载输入
            __syncthreads();
            q_s[v_idx] = float(q_[ind + v_idx]);
            w_raw_s[v_idx] = float(w_[ind + v_idx]);
            w_s[v_idx] = safe_exp(-safe_exp(w_raw_s[v_idx]));
            k_s[v_idx] = float(k_[ind + v_idx]);
            a_s[v_idx] = float(a_[ind + v_idx]);
            b_s[v_idx] = float(b_[ind + v_idx]);
            dy_s[v_idx] = float(dy_[ind + v_idx]);
            v_s[v_idx] = float(v_[ind + v_idx]);
            __syncthreads();
            
            // 获取前一时刻和当前时刻的状态
            float prev_state[256], curr_state[256];
            for (int k = 0; k < C; k++) {
                prev_state[k] = seg_states[st][k];
                curr_state[k] = seg_states[st + 1][k];
            }
            
            // 计算 sa = prev_state @ a
            float sa = 0.0f;
            for (int k = 0; k < C; k++) {
                sa += prev_state[k] * a_s[k];
            }
            sa_s[v_idx] = sa;
            __syncthreads();
            
            // dstate_curr = dstate + q * dy
            float dstate_curr[256];
            for (int k = 0; k < C; k++) {
                dstate_curr[k] = dstate[k] + q_s[k] * dy_s[v_idx];
            }
            
            // === dq: 需要 reduction ===
            __syncthreads();
            for (int k = 0; k < C; k++) {
                dq_buf[k * C + v_idx] = curr_state[k] * dy_s[v_idx];
            }
            __syncthreads();
            float dq_val = 0.0f;
            for (int v = 0; v < C; v++) {
                dq_val += dq_buf[v_idx * C + v];
            }
            dq_[ind + v_idx] = F(dq_val);
            
            // === dw: 需要 reduction ===
            __syncthreads();
            for (int k = 0; k < C; k++) {
                dq_buf[k * C + v_idx] = -dstate_curr[k] * prev_state[k];
            }
            __syncthreads();
            float dw_val = 0.0f;
            for (int v = 0; v < C; v++) {
                dw_val += dq_buf[v_idx * C + v];
            }
            dw_[ind + v_idx] = F(dw_val * w_s[v_idx] * safe_exp(w_raw_s[v_idx]));
            
            // === dv: 不需要 reduction ===
            float dv_val = 0.0f;
            for (int k = 0; k < C; k++) {
                dv_val += dstate_curr[k] * k_s[k];
            }
            dv_[ind + v_idx] = F(dv_val);
            
            // === dk: 需要 reduction ===
            __syncthreads();
            for (int k = 0; k < C; k++) {
                dq_buf[k * C + v_idx] = dstate_curr[k] * v_s[v_idx];
            }
            __syncthreads();
            float dk_val = 0.0f;
            for (int v = 0; v < C; v++) {
                dk_val += dq_buf[v_idx * C + v];
            }
            dk_[ind + v_idx] = F(dk_val);
            
            // === db: 需要 reduction ===
            __syncthreads();
            for (int k = 0; k < C; k++) {
                dq_buf[k * C + v_idx] = dstate_curr[k] * sa_s[v_idx];
            }
            __syncthreads();
            float db_val = 0.0f;
            for (int v = 0; v < C; v++) {
                db_val += dq_buf[v_idx * C + v];
            }
            db_[ind + v_idx] = F(db_val);
            
            // === dsa, da ===
            float dsa = 0.0f;
            for (int k = 0; k < C; k++) {
                dsa += dstate_curr[k] * b_s[k];
            }
            __syncthreads();
            dsa_s[v_idx] = dsa;
            __syncthreads();
            
            for (int k = 0; k < C; k++) {
                dq_buf[k * C + v_idx] = prev_state[k] * dsa_s[v_idx];
            }
            __syncthreads();
            float da_val = 0.0f;
            for (int v = 0; v < C; v++) {
                da_val += dq_buf[v_idx * C + v];
            }
            da_[ind + v_idx] = F(da_val);
            
            // === dstate 传播到前一时刻 ===
            for (int k = 0; k < C; k++) {
                dstate[k] = a_s[k] * dsa + dstate_curr[k] * w_s[k];
            }
        }
    }
    
    // 写回初始状态梯度
    for (int k = 0; k < C; k++) {
        ds_out_[state_base + k] = dstate[k];
    }
}

// ============================================================================
// C++ Template Wrappers
// ============================================================================

template <typename scalar_t>
void cuda_forward_infctx_template(int B, int T, int H, int C, scalar_t* w, scalar_t* q, scalar_t* k, scalar_t* v, scalar_t* a, scalar_t* b, float* s, scalar_t* y) {
    dim3 grid(H, B);
    dim3 block(C);
    forward_kernel_infctx<scalar_t><<<grid, block>>>(B, T, H, C, w, q, k, v, a, b, s, y);
}

template <typename scalar_t>
void cuda_backward_infctx_template(int B, int T, int H, int C, scalar_t* w, scalar_t* q, scalar_t* k, scalar_t* v, scalar_t* a, scalar_t* b, scalar_t* dy, float* s, float* ds_in, scalar_t* dw, scalar_t* dq, scalar_t* dk, scalar_t* dv, scalar_t* da, scalar_t* db, float* ds) {
    dim3 grid(H, B);
    dim3 block(C);
    size_t shared_mem_size = C * C * sizeof(float);  // 1 reduction buffer
    backward_kernel_recompute<scalar_t><<<grid, block, shared_mem_size>>>(B, T, H, C, w, q, k, v, a, b, dy, s, ds_in, dw, dq, dk, dv, da, db, ds);
}

// ============================================================================
// FP16/BF16/FP32 Functions
// ============================================================================

void cuda_forward_infctx_fp16(int B, int T, int H, int C, at::Half* w, at::Half* q, at::Half* k, at::Half* v, at::Half* a, at::Half* b, float* s, at::Half* y) { cuda_forward_infctx_template<at::Half>(B, T, H, C, w, q, k, v, a, b, s, y); }
void cuda_backward_infctx_fp16(int B, int T, int H, int C, at::Half* w, at::Half* q, at::Half* k, at::Half* v, at::Half* a, at::Half* b, at::Half* dy, float* s, float* ds_in, at::Half* dw, at::Half* dq, at::Half* dk, at::Half* dv, at::Half* da, at::Half* db, float* ds) { cuda_backward_infctx_template<at::Half>(B, T, H, C, w, q, k, v, a, b, dy, s, ds_in, dw, dq, dk, dv, da, db, ds); }

void cuda_forward_infctx_bf16(int B, int T, int H, int C, at::BFloat16* w, at::BFloat16* q, at::BFloat16* k, at::BFloat16* v, at::BFloat16* a, at::BFloat16* b, float* s, at::BFloat16* y) { cuda_forward_infctx_template<at::BFloat16>(B, T, H, C, w, q, k, v, a, b, s, y); }
void cuda_backward_infctx_bf16(int B, int T, int H, int C, at::BFloat16* w, at::BFloat16* q, at::BFloat16* k, at::BFloat16* v, at::BFloat16* a, at::BFloat16* b, at::BFloat16* dy, float* s, float* ds_in, at::BFloat16* dw, at::BFloat16* dq, at::BFloat16* dk, at::BFloat16* dv, at::BFloat16* da, at::BFloat16* db, float* ds) { cuda_backward_infctx_template<at::BFloat16>(B, T, H, C, w, q, k, v, a, b, dy, s, ds_in, dw, dq, dk, dv, da, db, ds); }

void cuda_forward_infctx_fp32(int B, int T, int H, int C, float* w, float* q, float* k, float* v, float* a, float* b, float* s, float* y) { cuda_forward_infctx_template<float>(B, T, H, C, w, q, k, v, a, b, s, y); }
void cuda_backward_infctx_fp32(int B, int T, int H, int C, float* w, float* q, float* k, float* v, float* a, float* b, float* dy, float* s, float* ds_in, float* dw, float* dq, float* dk, float* dv, float* da, float* db, float* ds) { cuda_backward_infctx_template<float>(B, T, H, C, w, q, k, v, a, b, dy, s, ds_in, dw, dq, dk, dv, da, db, ds); }
