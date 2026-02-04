#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

template <typename T>
void swiglu_cpu(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float gate_val = llaisys::utils::cast<float>(gate[i]);
        float up_val = llaisys::utils::cast<float>(up[i]);
        
        // SwiGLU: out = up * (gate / (1 + exp(-gate)))
        // This is equivalent to: out = up * gate * sigmoid(gate)
        float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
        float result = up_val * gate_val * sigmoid_gate;
        
        out[i] = llaisys::utils::cast<T>(result);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");
    CHECK_SAME_SHAPE(out->shape(), gate->shape());
    CHECK_SAME_SHAPE(gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype());
    CHECK_SAME_DTYPE(gate->dtype(), up->dtype());
    
    size_t numel = out->numel();
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_cpu(reinterpret_cast<float *>(out->data()),
                             reinterpret_cast<const float *>(gate->data()),
                             reinterpret_cast<const float *>(up->data()),
                             numel);
        case LLAISYS_DTYPE_BF16:
            return swiglu_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                             reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
                             reinterpret_cast<const llaisys::bf16_t *>(up->data()),
                             numel);
        case LLAISYS_DTYPE_F16:
            return swiglu_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                             reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
                             reinterpret_cast<const llaisys::fp16_t *>(up->data()),
                             numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
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
