#pragma once

namespace lynx {

// Type-erased RAII wrapper for opaque GPU state pointers.
// Replaces raw void* + manual new/delete pattern across operator classes.
// The concrete GPU state type (GPUEigenState, GPUHamiltonianState, etc.)
// is defined in the .cu file; the header only sees this wrapper.
class GPUStatePtr {
public:
    using Deleter = void(*)(void*);

    GPUStatePtr() = default;
    ~GPUStatePtr() { reset(); }

    GPUStatePtr(GPUStatePtr&& o) noexcept
        : ptr_(o.ptr_), deleter_(o.deleter_) { o.ptr_ = nullptr; }

    GPUStatePtr& operator=(GPUStatePtr&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            deleter_ = o.deleter_;
            o.ptr_ = nullptr;
        }
        return *this;
    }

    GPUStatePtr(const GPUStatePtr&) = delete;
    GPUStatePtr& operator=(const GPUStatePtr&) = delete;

    // Take ownership of a new pointer with a type-erased deleter.
    template<typename T>
    void reset(T* p) {
        reset();
        ptr_ = p;
        deleter_ = [](void* pp) { delete static_cast<T*>(pp); };
    }

    // Release ownership and destroy the held object.
    void reset() {
        if (ptr_ && deleter_) deleter_(ptr_);
        ptr_ = nullptr;
        deleter_ = nullptr;
    }

    // Access the held pointer, cast to the concrete type.
    template<typename T> T* as() { return static_cast<T*>(ptr_); }
    template<typename T> const T* as() const { return static_cast<const T*>(ptr_); }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    void* ptr_ = nullptr;
    Deleter deleter_ = nullptr;
};

} // namespace lynx
