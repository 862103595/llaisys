#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"
#include <memory>

// 结构体定义放在实现文件中，不暴露给外部
struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::qwen2::Qwen2> model;
};

__C {

struct LlaisysQwen2Model* llaisysQwen2ModelCreate(const LlaisysQwen2Meta* meta,
                                                   llaisysDeviceType_t device,
                                                   int* device_ids,
                                                   int ndevice) {
    auto impl = std::make_unique<llaisys::models::qwen2::Qwen2>(
        meta, device, device_ids, static_cast<size_t>(ndevice));
    return new LlaisysQwen2Model{std::move(impl)};
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model) {
    delete model;
}

struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model* model) {
    if (model == nullptr || model->model == nullptr) return nullptr;
    return model->model->getWeights();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model* model,
                               int64_t* token_ids,
                               size_t ntoken) {
    if (model == nullptr || model->model == nullptr) return -1;
    return model->model->infer(token_ids, ntoken);
}

int64_t llaisysQwen2ModelInferWithCache(struct LlaisysQwen2Model* model,
                                        int64_t* token_ids,
                                        size_t ntoken,
                                        size_t pos_offset) {
    if (model == nullptr || model->model == nullptr) return -1;
    return model->model->inferWithCache(token_ids, ntoken, pos_offset);
}

void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model* model) {
    if (model == nullptr || model->model == nullptr) return;
    model->model->resetCache();
}

size_t llaisysQwen2ModelGetCacheLen(struct LlaisysQwen2Model* model) {
    if (model == nullptr || model->model == nullptr) return 0;
    return model->model->getCacheLen();
}

}
