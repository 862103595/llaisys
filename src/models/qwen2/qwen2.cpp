#include "qwen2.hpp"

#include <array>
#include <vector>
#include <cmath>
#include <cstring>
#include "llaisys.h"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../tensor/tensor.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
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
    weights_.out_embed = tensorCreate(
        std::array<size_t, 2>{meta_.voc, meta_.hs}.data(),  // lm_head: [vocab_size, hidden_size]
        2, 
        meta_.dtype, 
        device_, 
        device_id
    );

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
            std::array<size_t, 2>{q_dim, meta_.hs}.data(),  // [out_features, in_features]
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
            std::array<size_t, 2>{kv_dim, meta_.hs}.data(),  // [out_features, in_features]
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
            std::array<size_t, 2>{kv_dim, meta_.hs}.data(),  // [out_features, in_features]
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
            std::array<size_t, 2>{meta_.hs, q_dim}.data(),  // [out_features, in_features]
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
            std::array<size_t, 2>{mlp_dim, meta_.hs}.data(),  // [out_features, in_features]
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.mlp_up_w[i] = tensorCreate(
            std::array<size_t, 2>{mlp_dim, meta_.hs}.data(),  // [out_features, in_features]
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
        weights_.mlp_down_w[i] = tensorCreate(
            std::array<size_t, 2>{meta_.hs, mlp_dim}.data(),  // [out_features, in_features]
            2, 
            meta_.dtype, 
            device_, 
            device_id
        );
    }
    
    // Initialize KV cache
    initCache();
}

Qwen2::~Qwen2() {}

LlaisysQwen2Weights* Qwen2::getWeights() {
    return &weights_;
}

void Qwen2::initCache() {
    kv_cache_.resize(meta_.nlayer);
    for (size_t i = 0; i < meta_.nlayer; i++) {
        kv_cache_[i].k_cache = nullptr;
        kv_cache_[i].v_cache = nullptr;
        kv_cache_[i].cached_len = 0;
    }
}

void Qwen2::resetCache() {
    for (size_t i = 0; i < meta_.nlayer; i++) {
        kv_cache_[i].k_cache = nullptr;
        kv_cache_[i].v_cache = nullptr;
        kv_cache_[i].cached_len = 0;
    }
}

std::pair<tensor_t, tensor_t> Qwen2::updateCache(size_t layer_idx, tensor_t new_k, tensor_t new_v) {
    const auto device_id = (device_ids_.empty()) ? 0 : device_ids_[0];
    auto& cache = kv_cache_[layer_idx];
    
    size_t new_len = new_k->shape()[0];
    size_t nkvh = meta_.nkvh;
    size_t head_dim = meta_.di;
    
    if (cache.cached_len == 0) {
        // First time: just store the new K/V
        cache.k_cache = Tensor::create(
            std::vector<size_t>{new_len, nkvh, head_dim},
            meta_.dtype,
            device_,
            device_id
        );
        cache.v_cache = Tensor::create(
            std::vector<size_t>{new_len, nkvh, head_dim},
            meta_.dtype,
            device_,
            device_id
        );
        
        // Copy data
        size_t bytes = new_len * nkvh * head_dim * 2;  // BF16 = 2 bytes
        std::memcpy(cache.k_cache->data(), new_k->data(), bytes);
        std::memcpy(cache.v_cache->data(), new_v->data(), bytes);
        cache.cached_len = new_len;
        
        return {cache.k_cache, cache.v_cache};
    } else {
        // Append new K/V to existing cache
        size_t total_len = cache.cached_len + new_len;
        
        tensor_t new_k_cache = Tensor::create(
            std::vector<size_t>{total_len, nkvh, head_dim},
            meta_.dtype,
            device_,
            device_id
        );
        tensor_t new_v_cache = Tensor::create(
            std::vector<size_t>{total_len, nkvh, head_dim},
            meta_.dtype,
            device_,
            device_id
        );
        
        size_t elem_size = 2;  // BF16 = 2 bytes
        size_t cached_bytes = cache.cached_len * nkvh * head_dim * elem_size;
        size_t new_bytes = new_len * nkvh * head_dim * elem_size;
        
        // Copy cached data
        std::memcpy(new_k_cache->data(), cache.k_cache->data(), cached_bytes);
        std::memcpy(new_v_cache->data(), cache.v_cache->data(), cached_bytes);
        
        // Copy new data
        std::memcpy(
            reinterpret_cast<uint8_t*>(new_k_cache->data()) + cached_bytes,
            new_k->data(),
            new_bytes
        );
        std::memcpy(
            reinterpret_cast<uint8_t*>(new_v_cache->data()) + cached_bytes,
            new_v->data(),
            new_bytes
        );
        
        // Update cache
        cache.k_cache = new_k_cache;
        cache.v_cache = new_v_cache;
        cache.cached_len = total_len;
        
        return {cache.k_cache, cache.v_cache};
    }
}

int64_t Qwen2::infer(int64_t* token_ids, size_t ntoken) {
    // Reset cache for prefill mode
    resetCache();
    return inferWithCache(token_ids, ntoken, 0);
}

int64_t Qwen2::inferWithCache(int64_t* token_ids, size_t ntoken, size_t pos_offset) {
    const auto device_id = (device_ids_.empty()) ? 0 : device_ids_[0];
    
    tensor_t input_ids = Tensor::create(
        std::vector<size_t>{ntoken}, 
        LLAISYS_DTYPE_I64,
        device_, 
        device_id
    );
    input_ids->load(token_ids);
    
    tensor_t x = Tensor::create(
        std::vector<size_t>{ntoken, meta_.hs}, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::embedding(x, input_ids, weights_.in_embed->tensor);
    
    for(size_t i = 0; i < meta_.nlayer; i++) {
        tensor_t x_norm = Tensor::create(
            std::vector<size_t>{ntoken, meta_.hs}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::rms_norm(x_norm, x, weights_.attn_norm_w[i]->tensor, meta_.epsilon);
        
        // Compute Q, K, V for new tokens only
        tensor_t q = Tensor::create(
            std::vector<size_t>{ntoken, meta_.nh * meta_.dh}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(q, x_norm, weights_.attn_q_w[i]->tensor, weights_.attn_q_b[i]->tensor);
        
        tensor_t k = Tensor::create(
            std::vector<size_t>{ntoken, meta_.nkvh * meta_.di}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(k, x_norm, weights_.attn_k_w[i]->tensor, weights_.attn_k_b[i]->tensor);
        
        tensor_t v = Tensor::create(
            std::vector<size_t>{ntoken, meta_.nkvh * meta_.di}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(v, x_norm, weights_.attn_v_w[i]->tensor, weights_.attn_v_b[i]->tensor);

        // Position IDs for new tokens (with offset)
        tensor_t pos_ids = Tensor::create(
            std::vector<size_t>{ntoken}, 
            LLAISYS_DTYPE_I64, 
            device_, 
            device_id
        );
        std::vector<int64_t> pos_ids_vec(ntoken);
        for (size_t j = 0; j < ntoken; j++) {
            pos_ids_vec[j] = static_cast<int64_t>(pos_offset + j);
        }
        pos_ids->load(pos_ids_vec.data());
        
        // Reshape q/k/v to 3D: [ntoken, nh*dh] -> [ntoken, nh, dh]
        tensor_t q_3d = q->view(std::vector<size_t>{ntoken, meta_.nh, meta_.dh});
        tensor_t k_3d = k->view(std::vector<size_t>{ntoken, meta_.nkvh, meta_.di});
        tensor_t v_3d = v->view(std::vector<size_t>{ntoken, meta_.nkvh, meta_.di});
        
        // Apply RoPE to new tokens
        ops::rope(q_3d, q_3d, pos_ids, meta_.theta);
        ops::rope(k_3d, k_3d, pos_ids, meta_.theta);
        
        // Update KV cache and get full K/V
        auto [full_k, full_v] = updateCache(i, k_3d, v_3d);
    
        // Self-attention with full K/V cache
        tensor_t attn_val = Tensor::create(
            std::vector<size_t>{ntoken, meta_.nh, meta_.dh}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
        ops::self_attention(attn_val, q_3d, full_k, full_v, scale);
        
        // Reshape attn_val back to 2D: [ntoken, nh, dh] -> [ntoken, nh*dh]
        tensor_t attn_val_2d = attn_val->view(std::vector<size_t>{ntoken, meta_.nh * meta_.dh});
        tensor_t x_attn = Tensor::create(
            std::vector<size_t>{ntoken, meta_.hs}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(x_attn, attn_val_2d, weights_.attn_o_w[i]->tensor, nullptr);
        ops::add(x, x, x_attn);
        
        // MLP: post_attention_layernorm -> gate_proj/up_proj -> swiglu -> down_proj
        tensor_t x_mlp_norm = Tensor::create(
            std::vector<size_t>{ntoken, meta_.hs}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::rms_norm(x_mlp_norm, x, weights_.mlp_norm_w[i]->tensor, meta_.epsilon);
        
        tensor_t gate = Tensor::create(
            std::vector<size_t>{ntoken, meta_.im}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        tensor_t up = Tensor::create(
            std::vector<size_t>{ntoken, meta_.im}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(gate, x_mlp_norm, weights_.mlp_gate_w[i]->tensor, nullptr);
        ops::linear(up, x_mlp_norm, weights_.mlp_up_w[i]->tensor, nullptr);
        
        tensor_t swiglu_out = Tensor::create(
            std::vector<size_t>{ntoken, meta_.im}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::swiglu(swiglu_out, gate, up);
        
        tensor_t x_mlp = Tensor::create(
            std::vector<size_t>{ntoken, meta_.hs}, 
            meta_.dtype, 
            device_, 
            device_id
        );
        ops::linear(x_mlp, swiglu_out, weights_.mlp_down_w[i]->tensor, nullptr);
        ops::add(x, x, x_mlp);
    }
    
    tensor_t output_norm = Tensor::create(
        std::vector<size_t>{ntoken, meta_.hs}, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::rms_norm(output_norm, x, weights_.out_norm_w->tensor, meta_.epsilon);
    tensor_t output = Tensor::create(
        std::vector<size_t>{ntoken, meta_.voc}, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::linear(output, output_norm, weights_.out_embed->tensor, nullptr);
    
    // Get logits for the last token: output[-1, :] -> [vocab_size]
    tensor_t logits = output->slice(0, ntoken - 1, ntoken);  // [1, vocab_size]
    tensor_t logits_1d = logits->view(std::vector<size_t>{meta_.voc});  // [vocab_size]
    
    tensor_t max_idx = Tensor::create(
        std::vector<size_t>{1}, 
        LLAISYS_DTYPE_I64, 
        device_, 
        device_id
    );
    tensor_t max_val = Tensor::create(
        std::vector<size_t>{1}, 
        meta_.dtype, 
        device_, 
        device_id
    );
    ops::argmax(max_idx, max_val, logits_1d);
    int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx->data());
    return max_idx_data[0];
}

}  // namespace qwen2
}  // namespace models
}  // namespace llaisys
