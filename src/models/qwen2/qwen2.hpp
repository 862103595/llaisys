#pragma once

#include <vector>
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../../tensor/tensor.hpp"

namespace llaisys {
namespace models {
namespace qwen2 {

// KV Cache for each layer
struct KVCache {
    tensor_t k_cache;  // [cached_len, nkvh, head_dim]
    tensor_t v_cache;  // [cached_len, nkvh, head_dim]
    size_t cached_len; // Number of cached tokens
    
    KVCache() : k_cache(nullptr), v_cache(nullptr), cached_len(0) {}
};

class Qwen2 {
public:
    Qwen2(const LlaisysQwen2Meta* meta,
          llaisysDeviceType_t device,
          const int* device_ids,
          size_t ndevice);
    ~Qwen2();

    LlaisysQwen2Weights* getWeights();
    
    // Original infer (prefill mode - processes all tokens, resets cache)
    int64_t infer(int64_t* token_ids, size_t ntoken);
    
    // Incremental infer (decode mode - uses KV cache)
    int64_t inferWithCache(int64_t* token_ids, size_t ntoken, size_t pos_offset);
    
    // Reset KV cache
    void resetCache();
    
    // Get current cache length
    size_t getCacheLen() const { return kv_cache_.empty() ? 0 : kv_cache_[0].cached_len; }

private:
    LlaisysQwen2Meta meta_;
    LlaisysQwen2Weights weights_;

    llaisysDeviceType_t device_;

    std::vector<int> device_ids_;
    size_t ndevice_;
    
    // KV Cache for each layer
    std::vector<KVCache> kv_cache_;
    
    // Initialize KV cache
    void initCache();
    
    // Append new K/V to cache and return full K/V
    std::pair<tensor_t, tensor_t> updateCache(size_t layer_idx, tensor_t new_k, tensor_t new_v);
};

}  // namespace qwen2
}  // namespace models
}  // namespace llaisys
