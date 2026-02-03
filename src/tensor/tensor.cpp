#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const size_t ndim_ = ndim();
    if (ndim_ <= 1) {
        return true;  // 0维和1维tensor总是contiguous
    }
    
    const auto& strides_ = strides();
    const auto& shape_ = shape();
    
    // 检查最后一个维度的stride是否为1（contiguous tensor的要求）
    if (strides_[ndim_ - 1] != 1) {
        return false;
    }
    
    // 从后往前检查：strides[i] == strides[i+1] * shape[i+1]
    for (size_t i = ndim_ - 1; i > 0; i--) {
        if (strides_[i - 1] != strides_[i] * static_cast<ptrdiff_t>(shape_[i])) {
            return false;
        }
    }
    
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t ndim_ = ndim();
    CHECK_ARGUMENT(order.size() == ndim_, "order size must be equal to ndim");
    
    // 验证order是否是有效的排列：检查索引范围和重复
    std::vector<bool> used(ndim_, false);
    for (size_t i = 0; i < ndim_; i++) {
        CHECK_ARGUMENT(order[i] < ndim_, "order index out of range");
        CHECK_ARGUMENT(!used[order[i]], "order contains duplicate indices");
        used[order[i]] = true;
    }
    
    // 缓存引用，避免重复调用
    const auto& shape_ = shape();
    const auto& strides_ = strides();
    
    // 重新排列shape和strides
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        new_shape[i] = shape_[order[i]];
        new_strides[i] = strides_[order[i]];
    }
    
    return std::shared_ptr<Tensor>(new Tensor(TensorMeta{dtype(), new_shape, new_strides}, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(isContiguous(), "tensor is not contiguous");
    
    // 检查元素总数是否匹配
    size_t total_elems = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(total_elems == numel(), "total elements must be equal to numel");

    // 计算新的strides（从后往前，C-style row-major）
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i > 0; i--) {
        new_strides[i - 1] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i - 1]);
    }
    
    return std::shared_ptr<Tensor>(new Tensor(TensorMeta{dtype(), shape, new_strides}, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    const size_t ndim_ = ndim();
    CHECK_ARGUMENT(dim < ndim_, "dimension index out of range");
    
    const auto& shape_ = shape();
    const auto& strides_ = strides();
    
    // 验证范围
    CHECK_ARGUMENT(start <= end && end <= shape_[dim], "invalid slice range");
    
    // 计算新的shape和offset
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = end - start;
    
    // 计算新的offset：需要加上 start * stride[dim] 的字节偏移
    size_t element_offset = start * strides_[dim];
    size_t new_offset = _offset + element_offset * elementSize();
    
    // strides保持不变
    std::vector<ptrdiff_t> new_strides = strides_;
    
    return std::shared_ptr<Tensor>(
        new Tensor(TensorMeta{dtype(), new_shape, new_strides}, _storage, new_offset)
    );
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    
    llaisysMemcpyKind_t kind = this->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2D : LLAISYS_MEMCPY_D2D;
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        this->numel() * this->elementSize(),
        kind
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
