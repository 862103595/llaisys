#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template <typename T>
void linear_cpu(T *out, const T *in, const T *weight, const T *bias,
                size_t batch_size, size_t in_features, size_t out_features) {
    for (size_t i = 0; i < batch_size; i++) {
        const T *in_row = in + i * in_features;
        T *out_row = out + i * out_features;

        for (size_t j = 0; j < out_features; j++) {
            // Compute dot product: sum_k(in[i, k] * weight[j, k])
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; k++) {
                float in_val = llaisys::utils::cast<float>(in_row[k]);
                float w_val = llaisys::utils::cast<float>(weight[j * in_features + k]);
                sum += in_val * w_val;
            }

            // Add bias if provided
            if (bias != nullptr) {
                float b_val = llaisys::utils::cast<float>(bias[j]);
                sum += b_val;
            }

            out_row[j] = llaisys::utils::cast<T>(sum);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out, in, and weight must be contiguous.");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }
    ASSERT(in->ndim() == 2, "Linear: input must be 2D [batch_size, in_features].");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D [out_features, in_features].");
    if (bias != nullptr) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D [out_features].");
    }

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    ASSERT(weight->shape()[1] == in_features, "Linear: weight shape mismatch.");
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == out_features,
           "Linear: output shape mismatch.");
    if (bias != nullptr) {
        ASSERT(bias->shape()[0] == out_features, "Linear: bias size must match out_features.");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            return linear_cpu(reinterpret_cast<float *>(out->data()),
                              reinterpret_cast<const float *>(in->data()),
                              reinterpret_cast<const float *>(weight->data()),
                              bias != nullptr ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                              batch_size, in_features, out_features);
        case LLAISYS_DTYPE_BF16:
            return linear_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                              reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                              reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                              bias != nullptr ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr,
                              batch_size, in_features, out_features);
        case LLAISYS_DTYPE_F16:
            return linear_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                              reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                              reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                              bias != nullptr ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr,
                              batch_size, in_features, out_features);
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
