#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"

namespace llaisys {
namespace models {
namespace qwen2 {

class Qwen2 {
public:
    Qwen2(const LlaisysQwen2Meta* meta,
          llaisysDeviceType_t device,
          const int* device_ids,
          size_t ndevice);
    ~Qwen2();

    LlaisysQwen2Weights* getWeights();
    int64_t infer(int64_t* token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta meta_;
    LlaisysQwen2Weights weights_;

    llaisysDeviceType_t device_;

    std::vector<int> device_ids_;
    size_t ndevice_;

};

}  // namespace qwen2
}  // namespace models
}  // namespace llaisys
