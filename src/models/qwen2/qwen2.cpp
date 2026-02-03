#include "qwen2.hpp"

#include "llaisys.h"
#include "../../utils/check.hpp"

namespace llaisys {
namespace models {
namespace qwen2 {

Qwen2::Qwen2(const LlaisysQwen2Meta* meta,
             llaisysDeviceType_t device,
             const int* device_ids,
             size_t ndevice): meta_(*meta), device_(device), device_ids_(device_ids, device_ids + ndevice), ndevice_(ndevice) {
    CHECK_ARGUMENT(meta_.im > 0, "mlp intermediate size must be > 0");
    const auto device_id = (device_ids == nullptr) ? 0 : device_ids[0];
    const size_t q_dim = meta_.nh * meta_.dh;
    const size_t kv_dim = meta_.nkvh * meta_.di;
    const size_t mlp_dim = meta_.im;
    weights_.in_embed = tensorCreate(
        std::array<size_t, 2>{meta_.voc, meta_.hs}.data(), 
        2, 
        meta_.dtype, 
        device_, 
        device_id
    );
    weights_.out_embed = weights_.in_embed;

    weights_.out_norm_w = tensorCreate(
        std::array<size_t, 1>{meta_.hs}.data(), 
        1, 
        meta_.dtype, 
        device_, 
        device_id
    );
    
    weights_.attn_norm_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_q_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_q_b = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_k_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_k_b = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_v_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_v_b = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_o_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_norm_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_gate_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_up_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_down_w = new llaisysTensor_t[meta_.nlayer];

    for (size_t i = 0; i < meta_.nlayer; i++) {
        weights_.attn_norm_w[i] = tensorCreate(
            std::array<size_t, 1>{meta_.hs}.data(), 
            1, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_q_w[i] = tensorCreate(
            std::array<size_t, 2>{meta_.hs, q_dim}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_q_b[i] = tensorCreate(
            std::array<size_t, 1>{q_dim}.data(), 
            1, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_k_w[i] = tensorCreate(
            std::array<size_t, 2>{meta_.hs, kv_dim}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_k_b[i] = tensorCreate(
            std::array<size_t, 1>{kv_dim}.data(), 
            1, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_v_w[i] = tensorCreate(
            std::array<size_t, 2>{meta_.hs, kv_dim}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_v_b[i] = tensorCreate(
            std::array<size_t, 1>{kv_dim}.data(), 
            1, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.attn_o_w[i] = tensorCreate(
            std::array<size_t, 2>{q_dim, meta_.hs}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.mlp_norm_w[i] = tensorCreate(
            std::array<size_t, 1>{meta_.hs}.data(), 
            1, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.mlp_gate_w[i] = tensorCreate(
            std::array<size_t, 2>{meta_.hs, mlp_dim}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.mlp_up_w[i] = tensorCreate(
            std::array<size_t, 2>{meta_.hs, mlp_dim}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.mlp_down_w[i] = tensorCreate(
            std::array<size_t, 2>{mlp_dim, meta_.hs}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
    }

Qwen2::~Qwen2() {}

LlaisysQwen2Weights* Qwen2::getWeights() {
    return &weights_;
}

int64_t Qwen2::infer(int64_t* token_ids, size_t ntoken) {
    tensor_t input_ids = tensorCreate(
        std::array<size_t, 1>{ntoken}.data(), 
        1, 
        meta_.dtype, 
        device_, 
        device_id
    );
    input_ids->load(token_ids);
    tensor_t x = tensorCreate(
        std::array<size_t, 2>{ntoken, meta_.hs}.data(), 
        2, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::embedding(x, input_ids, weights_.in_embed);
    for(size_t i = 0; i < meta_.nlayer; i++) {
        tensor_t x_norm = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.hs}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::rms_norm(x_norm, x, weights_.attn_norm_w[i], meta_.epsilon);
        tensor_t q = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.nh * meta_.dh}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(q, x_norm, weights_.attn_q_w[i], weights_.attn_q_b[i]);
        tensor_t k = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.nkvh * meta_.di}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(k, x_norm, weights_.attn_k_w[i], weights_.attn_k_b[i]);
        tensor_t v = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.nkvh * meta_.di}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(v, x_norm, weights_.attn_v_w[i], weights_.attn_v_b[i]);

        tensor_t pos_ids = tensorCreate(
            std::array<size_t, 1>{ntoken}.data(), 
            1, 
            LLAISYS_DTYPE_I64, 
            device_, 
            device_id
        );
        std::vector<int64_t> pos_ids_vec(ntoken);
        for (size_t j = 0; j < ntoken; j++) {
            pos_ids_vec[j] = j;
        }
        pos_ids->load(pos_ids_vec.data());
        ops::rope(q, q, pos_ids, meta_.theta);
        ops::rope(k, k, pos_ids, meta_.theta);
    
        tensor_t attn_val = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.nh * meta_.dh}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::self_attention(attn_val, q, k, v, meta_.scale);
        tensor_t x_attn = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.hs}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(x_attn, attn_val, weights_.attn_o_w[i]);
        ops::add(x, x, x_attn);
        tensort_t x_mlp = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.hs}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        tensor_t gate = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.im}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        tensor_t up = tensorCreate(
            std::array<size_t, 2>{ntoken, meta_.im}.data(), 
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(gate, x, weights_.mlp_gate_w[i], weights_.mlp_gate_b[i]);
        ops::linear(up, x, weights_.mlp_up_w[i], weights_.mlp_up_b[i]);
        ops::swiglu(x_mlp, gate, up);
        ops::add(x, x, x_mlp);
    }
    tensor_t output_norm = tensorCreate(
        std::array<size_t, 2>{ntoken, meta_.hs}.data(), 
        2, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::rms_norm(output_norm, x, weights_.out_norm_w, meta_.epsilon);
    tensor_t output = tensorCreate(
        std::array<size_t, 2>{ntoken, meta_.voc}.data(), 
        2, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::linear(output, output_norm, weights_.out_embed);
    tensor_t logits = tensorCreate(
        std::array<size_t, 1>{meta_.voc}.data(), 
        1, 
        meta_.dtype, 
        device_, 
        device_id
    );  
    tensor_t max_idx = tensorCreate(
        std::array<size_t, 1>{1}.data(), 
        1, 
        LLAISYS_DTYPE_I64, 
        device_, 
        device_id
    );
    tensor_t max_val = tensorCreate(
        std::array<size_t, 1>{1}.data(), 
        1, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::argmax(max_idx, max_val, output);
    return max_idx->item();
}

}  // namespace qwen2
}  // namespace models
}  // namespace llaisys
