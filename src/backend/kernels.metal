//! Custom Metal kernels for performance optimization
//!
//! This file contains hand-optimized Metal compute shaders for operations
//! where kernel fusion and specialized algorithms provide significant speedups
//! over Candle's default implementations.
//!
//! Target: Achieve 95-110% of MLX performance on Apple Silicon

#include <metal_stdlib>
using namespace metal;

//==============================================================================
// PHASE 3: Fused LoRA Kernel
//==============================================================================
// 
// Fuses the LoRA forward pass into a single GPU kernel:
//   output = (input @ lora_a @ lora_b) * scaling
//
// Benefits:
//   - Single kernel launch (vs 2+ in Candle)
//   - No intermediate memory allocation
//   - Reduced memory bandwidth
//   - Expected speedup: 6-10x
//
//==============================================================================

/// Parameters for fused LoRA kernel
struct LoRAParams {
    uint batch_size;
    uint seq_len;
    uint in_features;
    uint rank;
    uint out_features;
    float scaling;
};

/// Fused LoRA forward pass kernel
///
/// Computes: output[b,s,o] = sum_r(sum_i(input[b,s,i] * lora_a[i,r]) * lora_b[r,o]) * scaling
///
/// This kernel fuses two matrix multiplications and a scaling operation into one,
/// avoiding intermediate memory allocations and reducing memory bandwidth.
///
/// Thread organization:
/// - Each thread computes one output element
/// - Grid dimensions: (batch_size, seq_len, out_features)
///
kernel void fused_lora_forward(
    const device float* input [[buffer(0)]],           // [batch, seq, in_features]
    const device float* lora_a [[buffer(1)]],          // [in_features, rank]
    const device float* lora_b [[buffer(2)]],          // [rank, out_features]
    device float* output [[buffer(3)]],                // [batch, seq, out_features]
    constant LoRAParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes output[batch, seq, out_feature]
    uint batch = gid.x;
    uint seq = gid.y;
    uint out_feature = gid.z;
    
    // Bounds check
    if (batch >= params.batch_size || seq >= params.seq_len || out_feature >= params.out_features) {
        return;
    }
    
    // Accumulator for the output value
    float result = 0.0f;
    
    // Fused computation: (input @ lora_a) @ lora_b
    // Loop over rank dimension (hidden dimension between two matmuls)
    for (uint r = 0; r < params.rank; r++) {
        // First matmul: input @ lora_a -> hidden[r]
        float hidden = 0.0f;
        for (uint i = 0; i < params.in_features; i++) {
            uint input_idx = batch * params.seq_len * params.in_features +
                           seq * params.in_features +
                           i;
            uint lora_a_idx = i * params.rank + r;
            hidden += input[input_idx] * lora_a[lora_a_idx];
        }
        
        // Second matmul: hidden @ lora_b -> output[out_feature]
        uint lora_b_idx = r * params.out_features + out_feature;
        result += hidden * lora_b[lora_b_idx];
    }
    
    // Apply scaling
    result *= params.scaling;
    
    // Write output
    uint output_idx = batch * params.seq_len * params.out_features +
                     seq * params.out_features +
                     out_feature;
    output[output_idx] = result;
}

/// Optimized fused LoRA kernel with improved memory access
///
/// This version optimizes memory access patterns while maintaining correctness.
/// Uses loop unrolling hints and better instruction scheduling.
///
/// Strategy:
/// - Maintain correct algorithm from naive version
/// - Optimize memory access patterns
/// - Use compiler hints for better code generation
/// - Future: Consider MPS (Metal Performance Shaders) for matmul
///
kernel void fused_lora_forward_tiled(
    const device float* input [[buffer(0)]],           // [batch, seq, in_features]
    const device float* lora_a [[buffer(1)]],          // [in_features, rank]
    const device float* lora_b [[buffer(2)]],          // [rank, out_features]
    device float* output [[buffer(3)]],                // [batch, seq, out_features]
    constant LoRAParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes output[batch, seq, out_feature]
    uint batch = gid.x;
    uint seq = gid.y;
    uint out_feature = gid.z;
    
    // Bounds check
    if (batch >= params.batch_size || seq >= params.seq_len || out_feature >= params.out_features) {
        return;
    }
    
    // Pre-calculate base indices for better instruction scheduling
    uint input_base = batch * params.seq_len * params.in_features + seq * params.in_features;
    
    // Accumulator for the output value
    float result = 0.0f;
    
    // Fused computation: (input @ lora_a) @ lora_b
    // Loop over rank dimension (hidden dimension between two matmuls)
    for (uint r = 0; r < params.rank; r++) {
        // First matmul: input @ lora_a -> hidden[r]
        float hidden = 0.0f;
        
        // Inner loop over in_features
        // Compiler will attempt to vectorize/unroll this
        for (uint i = 0; i < params.in_features; i++) {
            uint input_idx = input_base + i;
            uint lora_a_idx = i * params.rank + r;
            hidden += input[input_idx] * lora_a[lora_a_idx];
        }
        
        // Second matmul: hidden @ lora_b -> output[out_feature]
        uint lora_b_idx = r * params.out_features + out_feature;
        result += hidden * lora_b[lora_b_idx];
    }
    
    // Apply scaling
    result *= params.scaling;
    
    // Write output
    uint output_idx = batch * params.seq_len * params.out_features +
                     seq * params.out_features +
                     out_feature;
    output[output_idx] = result;
}

//==============================================================================
// PHASE 4: Layer Operation Kernels
//==============================================================================

/// Parameters for fused softmax kernel
struct SoftmaxParams {
    uint batch_size;
    uint seq_len;
    uint dim;  // Dimension over which to apply softmax
};

/// Fused softmax kernel with threadgroup reductions
///
/// Computes numerically stable softmax in a single kernel:
///   1. Find max(x) using parallel reduction
///   2. Compute exp(x - max) and sum in parallel
///   3. Divide by sum
///
/// This avoids multiple kernel launches and intermediate memory allocations.
///
/// Thread organization:
/// - Each threadgroup handles one softmax operation (one row)
/// - Threads cooperate to compute max and sum reductions
/// - Grid: (batch_size, seq_len, 1)
/// - Threadgroup: (256, 1, 1) for optimal reduction performance
///
kernel void fused_softmax(
    const device float* input [[buffer(0)]],           // [batch, seq, dim]
    device float* output [[buffer(1)]],                // [batch, seq, dim]
    constant SoftmaxParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint batch = gid.z;
    uint seq = gid.y;
    uint thread_id = tid.x;
    
    // Bounds check
    if (batch >= params.batch_size || seq >= params.seq_len) {
        return;
    }
    
    uint row_offset = batch * params.seq_len * params.dim + seq * params.dim;
    
    // STEP 1: Find maximum value using parallel reduction
    float thread_max = -INFINITY;
    uint tg_size_x = tg_size.x;
    
    // Each thread processes multiple elements if dim > threadgroup size
    for (uint i = thread_id; i < params.dim; i += tg_size_x) {
        float val = input[row_offset + i];
        thread_max = max(thread_max, val);
    }
    
    // Store in shared memory
    shared_data[thread_id] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find global max
    for (uint stride = tg_size_x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float max_val = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // STEP 2: Compute exp(x - max) and accumulate sum
    float thread_sum = 0.0f;
    
    for (uint i = thread_id; i < params.dim; i += tg_size_x) {
        float val = input[row_offset + i];
        float exp_val = exp(val - max_val);
        output[row_offset + i] = exp_val;  // Store temporarily
        thread_sum += exp_val;
    }
    
    // Store partial sums in shared memory
    shared_data[thread_id] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to compute total sum
    for (uint stride = tg_size_x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float sum_val = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // STEP 3: Divide by sum to get final softmax values
    for (uint i = thread_id; i < params.dim; i += tg_size_x) {
        output[row_offset + i] /= sum_val;
    }
}

//==============================================================================
// RMS Norm Kernel
//==============================================================================

/// Parameters for fused RMS norm kernel
struct RMSNormParams {
    uint batch_size;
    uint seq_len;
    uint dim;
    float eps;  // Small constant for numerical stability
};

/// Fused RMS normalization kernel
///
/// Computes: output = x / sqrt(mean(x^2) + eps)
///
/// Single kernel with parallel reduction for mean computation.
///
kernel void fused_rms_norm(
    const device float* input [[buffer(0)]],           // [batch, seq, dim]
    device float* output [[buffer(1)]],                // [batch, seq, dim]
    constant RMSNormParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint batch = gid.z;
    uint seq = gid.y;
    uint thread_id = tid.x;
    
    // Bounds check
    if (batch >= params.batch_size || seq >= params.seq_len) {
        return;
    }
    
    uint row_offset = batch * params.seq_len * params.dim + seq * params.dim;
    
    // STEP 1: Compute sum of squares using parallel reduction
    float thread_sum_sq = 0.0f;
    uint tg_size_x = tg_size.x;
    
    for (uint i = thread_id; i < params.dim; i += tg_size_x) {
        float val = input[row_offset + i];
        thread_sum_sq += val * val;
    }
    
    // Store in shared memory
    shared_data[thread_id] = thread_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tg_size_x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean_sq = shared_data[0] / float(params.dim);
    float rms = sqrt(mean_sq + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // STEP 2: Normalize
    for (uint i = thread_id; i < params.dim; i += tg_size_x) {
        output[row_offset + i] = input[row_offset + i] / rms;
    }
}

//==============================================================================
//==============================================================================
// PHASE 4: Layer Normalization (for Embeddings)
//==============================================================================
// 
// Standard Layer Normalization: normalize to mean=0, variance=1
//   normalized[i] = (input[i] - mean) / sqrt(variance + eps)
//
// Benefits:
//   - Enables Metal acceleration for embedding models (BERT, E5, MiniLM)
//   - Fused mean/variance computation
//   - Single-pass reduction
//   - Expected speedup: 5-10x over CPU fallback
//
//==============================================================================

/// Parameters for layer normalization kernel
struct LayerNormParams {
    uint batch_size;
    uint hidden_size;
    float eps;
};

/// Layer Normalization kernel
///
/// Computes: output[b,h] = (input[b,h] - mean[b]) / sqrt(variance[b] + eps)
///
/// This kernel normalizes each sequence in the batch independently.
/// Uses two passes:
/// 1. Compute mean and variance
/// 2. Normalize using computed statistics
///
/// Thread organization:
/// - Each thread processes one element in the hidden dimension
/// - Threads cooperate within a threadgroup to compute reduction (mean/variance)
///
kernel void layer_norm(
    const device float* input [[buffer(0)]],        // [batch, hidden_size]
    device float* output [[buffer(1)]],             // [batch, hidden_size]
    constant LayerNormParams& params [[buffer(2)]], // Parameters
    uint batch_idx [[threadgroup_position_in_grid]],
    uint hidden_idx [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    // Shared memory for reductions
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];
    
    const uint offset = batch_idx * params.hidden_size;
    const uint tid = hidden_idx;
    
    // Phase 1: Compute sum and sum of squares (for mean and variance)
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (uint h = tid; h < params.hidden_size; h += threadgroup_size) {
        float val = input[offset + h];
        sum += val;
        sq_sum += val * val;
    }
    
    shared_sum[tid] = sum;
    shared_sq_sum[tid] = sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to compute total sum and sum of squares
    for (uint stride = threadgroup_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 computes mean and variance
    threadgroup float mean;
    threadgroup float inv_std;
    
    if (tid == 0) {
        float total_sum = shared_sum[0];
        float total_sq_sum = shared_sq_sum[0];
        
        mean = total_sum / float(params.hidden_size);
        float variance = (total_sq_sum / float(params.hidden_size)) - (mean * mean);
        inv_std = rsqrt(variance + params.eps);  // 1 / sqrt(variance + eps)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Normalize using computed mean and variance
    for (uint h = tid; h < params.hidden_size; h += threadgroup_size) {
        float val = input[offset + h];
        output[offset + h] = (val - mean) * inv_std;
    }
}
//==============================================================================

