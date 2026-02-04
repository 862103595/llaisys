#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

template <typename T>
void rms_norm_cpu(T *out, const T *in, const T *weight, size_t batch_size, size_t hidden_dim, float eps) {
    for (size_t i = 0; i < batch_size; i++) {
        const T *in_row = in + i * hidden_dim;
        T *out_row = out + i * hidden_dim;

        // Compute mean of x^2 along the last dimension
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_dim; j++) {
            float val = llaisys::utils::cast<float>(in_row[j]);
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / static_cast<float>(hidden_dim);
        float rms = std::sqrt(mean_sq + eps);
        float inv_rms = 1.0f / rms;

        // Compute output: y = (x / rms) * weight
        for (size_t j = 0; j < hidden_dim; j++) {
            float x_val = llaisys::utils::cast<float>(in_row[j]);
            float w_val = llaisys::utils::cast<float>(weight[j]);
            float y_val = x_val * inv_rms * w_val;
            out_row[j] = llaisys::utils::cast<T>(y_val);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");
    ASSERT(in->ndim() == 2, "RMSNorm: input must be 2D [batch_size, hidden_dim].");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D [hidden_dim].");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());

    size_t batch_size = in->shape()[0];
    size_t hidden_dim = in->shape()[1];
    ASSERT(weight->shape()[0] == hidden_dim, "RMSNorm: weight size must match hidden_dim.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_cpu(reinterpret_cast<float *>(out->data()),
                                reinterpret_cast<const float *>(in->data()),
                                reinterpret_cast<const float *>(weight->data()),
                                batch_size, hidden_dim, eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                                reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                                reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                                batch_size, hidden_dim, eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                                reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                                reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                                batch_size, hidden_dim, eps);
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
