#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template <typename IndexT, typename DataT>
void embedding_cpu(DataT *out, const IndexT *index, const DataT *weight,
                   size_t num_indices, size_t embedding_dim) {
    for (size_t i = 0; i < num_indices; i++) {
        IndexT idx = index[i];
        const DataT *src = weight + idx * embedding_dim;
        DataT *dst = out + i * embedding_dim;
        for (size_t j = 0; j < embedding_dim; j++) {
            dst[j] = src[j];
        }
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D [vocab_size, embedding_dim].");

    size_t num_indices = index->numel();
    size_t embedding_dim = weight->shape().back();

    // Verify output shape: should be [*index.shape, embedding_dim]
    ASSERT(out->numel() == num_indices * embedding_dim, "Embedding: output shape mismatch.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // Dispatch based on index type and weight type
        auto dispatch_index = [&](auto index_ptr) {
            switch (weight->dtype()) {
            case LLAISYS_DTYPE_F32:
                return embedding_cpu(reinterpret_cast<float *>(out->data()), index_ptr,
                                     reinterpret_cast<const float *>(weight->data()),
                                     num_indices, embedding_dim);
            case LLAISYS_DTYPE_BF16:
                return embedding_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()), index_ptr,
                                     reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                                     num_indices, embedding_dim);
            case LLAISYS_DTYPE_F16:
                return embedding_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()), index_ptr,
                                     reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                                     num_indices, embedding_dim);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
            }
        };

        switch (index->dtype()) {
        case LLAISYS_DTYPE_I32:
            return dispatch_index(reinterpret_cast<const int32_t *>(index->data()));
        case LLAISYS_DTYPE_I64:
            return dispatch_index(reinterpret_cast<const int64_t *>(index->data()));
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(index->dtype());
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
