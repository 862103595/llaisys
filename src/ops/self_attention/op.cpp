#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <algorithm>

namespace llaisys::ops {

template <typename T>
void self_attention_cpu(T *attn_val, const T *q, const T *k, const T *v,
                       size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    // Allocate temporary buffers for attention computation
    // attn_weight: [nh, qlen, kvlen]
    std::vector<float> attn_weight(nh * qlen * kvlen);
    
    // Step 1: Compute attention weights: Q @ K^T * scale
    // Q: [qlen, nh, hd] -> view as [nh, qlen, hd]
    // K: [kvlen, nkvh, hd] -> view as [nkvh, kvlen, hd], then repeat to [nh, kvlen, hd]
    // attn_weight: [nh, qlen, kvlen]
    size_t heads_per_group = nh / nkvh;
    for (size_t h = 0; h < nh; h++) {
        size_t kvh = h / heads_per_group;  // GQA: repeat_interleave mapping
        for (size_t i = 0; i < qlen; i++) {
            for (size_t j = 0; j < kvlen; j++) {
                float sum = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    float q_val = llaisys::utils::cast<float>(q[(i * nh + h) * hd + d]);
                    float k_val = llaisys::utils::cast<float>(k[(j * nkvh + kvh) * hd + d]);
                    sum += q_val * k_val;
                }
                attn_weight[(h * qlen + i) * kvlen + j] = sum * scale;
            }
        }
    }
    
    // Step 2: Add causal mask (lower triangular mask)
    // torch.ones(L, S).tril(diagonal=S-L) creates a lower triangular mask
    // where positions (i, j) with j <= i + (S - L) are True
    int64_t diagonal = static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen);
    for (size_t h = 0; h < nh; h++) {
        for (size_t i = 0; i < qlen; i++) {
            for (size_t j = 0; j < kvlen; j++) {
                // Causal mask: mask positions where j > i + diagonal
                int64_t i64 = static_cast<int64_t>(i);
                int64_t j64 = static_cast<int64_t>(j);
                if (j64 > i64 + diagonal) {
                    attn_weight[(h * qlen + i) * kvlen + j] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    
    // Step 3: Softmax over the last dimension (kvlen)
    for (size_t h = 0; h < nh; h++) {
        for (size_t i = 0; i < qlen; i++) {
            float *row = &attn_weight[(h * qlen + i) * kvlen];
            
            // Find max for numerical stability
            float max_val = *std::max_element(row, row + kvlen);
            
            // Compute exp(x - max) and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                row[j] = std::exp(row[j] - max_val);
                sum_exp += row[j];
            }
            
            // Normalize
            for (size_t j = 0; j < kvlen; j++) {
                row[j] /= sum_exp;
            }
        }
    }
    
    // Step 4: Compute attn_val = attn_weight @ V
    // attn_weight: [nh, qlen, kvlen]
    // V: [kvlen, nkvh, hd] -> view as [nkvh, kvlen, hd], then repeat to [nh, kvlen, hd]
    // attn_val: [qlen, nh, hd]
    for (size_t h = 0; h < nh; h++) {
        size_t kvh = h / heads_per_group;  // GQA: repeat_interleave mapping
        for (size_t i = 0; i < qlen; i++) {
            for (size_t d = 0; d < hd; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    float attn_w = attn_weight[(h * qlen + i) * kvlen + j];
                    float v_val = llaisys::utils::cast<float>(v[(j * nkvh + kvh) * hd + d]);
                    sum += attn_w * v_val;
                }
                attn_val[(i * nh + h) * hd + d] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D [qlen, nh, hd].");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D [kvlen, nkvh, hd].");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D [kvlen, nkvh, hd].");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D [qlen, nh, hd].");
    
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    
    ASSERT(q->shape()[2] == hd && k->shape()[2] == hd && v->shape()[2] == hd,
           "SelfAttention: head_dim mismatch.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "SelfAttention: attn_val shape mismatch.");
    ASSERT(nh % nkvh == 0, "SelfAttention: nh must be divisible by nkvh.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(q->dtype(), k->dtype());
    CHECK_SAME_DTYPE(k->dtype(), v->dtype());
    
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (q->dtype()) {
        case LLAISYS_DTYPE_F32:
            return self_attention_cpu(reinterpret_cast<float *>(attn_val->data()),
                                     reinterpret_cast<const float *>(q->data()),
                                     reinterpret_cast<const float *>(k->data()),
                                     reinterpret_cast<const float *>(v->data()),
                                     qlen, kvlen, nh, nkvh, hd, scale);
        case LLAISYS_DTYPE_BF16:
            return self_attention_cpu(reinterpret_cast<llaisys::bf16_t *>(attn_val->data()),
                                     reinterpret_cast<const llaisys::bf16_t *>(q->data()),
                                     reinterpret_cast<const llaisys::bf16_t *>(k->data()),
                                     reinterpret_cast<const llaisys::bf16_t *>(v->data()),
                                     qlen, kvlen, nh, nkvh, hd, scale);
        case LLAISYS_DTYPE_F16:
            return self_attention_cpu(reinterpret_cast<llaisys::fp16_t *>(attn_val->data()),
                                     reinterpret_cast<const llaisys::fp16_t *>(q->data()),
                                     reinterpret_cast<const llaisys::fp16_t *>(k->data()),
                                     reinterpret_cast<const llaisys::fp16_t *>(v->data()),
                                     qlen, kvlen, nh, nkvh, hd, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
        }
    }
    
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    
    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
