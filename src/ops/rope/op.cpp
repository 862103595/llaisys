#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

template <typename T>
void rope_cpu(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    
    for (size_t i = 0; i < seq_len; i++) {
        int64_t pos = pos_ids[i];
        float pos_f = static_cast<float>(pos);
        
        for (size_t h = 0; h < n_heads; h++) {
            const T *in_head = in + (i * n_heads + h) * head_dim;
            T *out_head = out + (i * n_heads + h) * head_dim;
            
            // Apply RoPE for each dimension pair
            for (size_t j = 0; j < half_dim; j++) {
                // Compute angle: phi = pos / (theta^(2j/head_dim))
                // Use division instead of negative exponent to match PyTorch precision
                float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
                float freq = std::pow(theta, exponent);
                float angle = pos_f / freq;
                
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // Get a and b values
                float a_val = llaisys::utils::cast<float>(in_head[j]);
                float b_val = llaisys::utils::cast<float>(in_head[j + half_dim]);
                
                // Apply rotation:
                // a' = a * cos - b * sin
                // b' = b * cos + a * sin
                float a_out = a_val * cos_val - b_val * sin_val;
                float b_out = b_val * cos_val + a_val * sin_val;
                
                out_head[j] = llaisys::utils::cast<T>(a_out);
                out_head[j + half_dim] = llaisys::utils::cast<T>(b_out);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");
    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [seq_len, n_heads, head_dim].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seq_len].");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");
    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids length must match seq_len.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rope_cpu(reinterpret_cast<float *>(out->data()),
                          reinterpret_cast<const float *>(in->data()),
                          reinterpret_cast<const int64_t *>(pos_ids->data()),
                          seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                          reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                          reinterpret_cast<const int64_t *>(pos_ids->data()),
                          seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_F16:
            return rope_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                          reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                          reinterpret_cast<const int64_t *>(pos_ids->data()),
                          seq_len, n_heads, head_dim, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        TO_BE_IMPLEMENTED();
        return;
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
