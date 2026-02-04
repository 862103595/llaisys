#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template <typename T>
void argmax_cpu(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        return;
    }
    
    T max_value = vals[0];
    size_t max_index = 0;
    
    for (size_t i = 1; i < numel; i++) {
        float val = llaisys::utils::cast<float>(vals[i]);
        float max_val_f = llaisys::utils::cast<float>(max_value);
        if (val > max_val_f) {
            max_value = vals[i];
            max_index = i;
        }
    }
    
    *max_idx = static_cast<int64_t>(max_index);
    *max_val = max_value;
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->isContiguous(), "Argmax: vals must be contiguous.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax: max_idx and max_val must be contiguous.");
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D for now.");
    ASSERT(max_idx->ndim() == 1 && max_idx->numel() == 1,
           "Argmax: max_idx must be 1D with single element.");
    ASSERT(max_val->ndim() == 1 && max_val->numel() == 1,
           "Argmax: max_val must be 1D with single element.");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    
    size_t numel = vals->numel();
    
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                             reinterpret_cast<float *>(max_val->data()),
                             reinterpret_cast<const float *>(vals->data()),
                             numel);
        case LLAISYS_DTYPE_BF16:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                             reinterpret_cast<llaisys::bf16_t *>(max_val->data()),
                             reinterpret_cast<const llaisys::bf16_t *>(vals->data()),
                             numel);
        case LLAISYS_DTYPE_F16:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                             reinterpret_cast<llaisys::fp16_t *>(max_val->data()),
                             reinterpret_cast<const llaisys::fp16_t *>(vals->data()),
                             numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }
    
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    
#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        TO_BE_IMPLEMENTED();
        return;
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
