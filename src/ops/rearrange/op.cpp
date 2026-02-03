#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template <typename T>
void rearrange_cpu(T *out, const T *in, const std::vector<size_t> &out_shape,
                   const std::vector<ptrdiff_t> &out_strides,
                   const std::vector<ptrdiff_t> &in_strides, size_t ndim) {
    if (ndim == 0) {
        // Scalar case
        *out = *in;
        return;
    }
    
    // Recursive helper to iterate through all dimensions
    std::vector<size_t> indices(ndim, 0);
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= out_shape[i];
    }
    
    for (size_t flat_idx = 0; flat_idx < total_elements; flat_idx++) {
        // Compute multi-dimensional indices from flat index
        size_t remaining = flat_idx;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; i--) {
            indices[i] = remaining % out_shape[i];
            remaining /= out_shape[i];
        }
        
        // Compute input and output offsets
        size_t in_offset = 0;
        size_t out_offset = 0;
        for (size_t i = 0; i < ndim; i++) {
            in_offset += indices[i] * in_strides[i];
            out_offset += indices[i] * out_strides[i];
        }
        
        out[out_offset] = in[in_offset];
    }
}

void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    size_t ndim = out->ndim();
    const auto &out_shape = out->shape();
    const auto &out_strides = out->strides();
    const auto &in_strides = in->strides();
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rearrange_cpu(reinterpret_cast<float *>(out->data()),
                               reinterpret_cast<const float *>(in->data()),
                               out_shape, out_strides, in_strides, ndim);
        case LLAISYS_DTYPE_BF16:
            return rearrange_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                               reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                               out_shape, out_strides, in_strides, ndim);
        case LLAISYS_DTYPE_F16:
            return rearrange_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                               reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                               out_shape, out_strides, in_strides, ndim);
        case LLAISYS_DTYPE_I32:
            return rearrange_cpu(reinterpret_cast<int32_t *>(out->data()),
                               reinterpret_cast<const int32_t *>(in->data()),
                               out_shape, out_strides, in_strides, ndim);
        case LLAISYS_DTYPE_I64:
            return rearrange_cpu(reinterpret_cast<int64_t *>(out->data()),
                               reinterpret_cast<const int64_t *>(in->data()),
                               out_shape, out_strides, in_strides, ndim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
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
