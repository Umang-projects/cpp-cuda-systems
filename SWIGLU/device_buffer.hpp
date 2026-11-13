#pragma once
#include <cstddef>
#include <cuda_runtime.h>

template<typename T>
struct TensorView2D{
    T* ptr_;
    int rows_;
    int cols_;

    __host__ __device__ inline T& operator()(int r,int c) const {
        return ptr_[r*cols_+c];
    }
};

template<typename T>
struct DeviceBuffer{
    T* ptr_=nullptr;
    size_t size_=0;

    DeviceBuffer(size_t n):size_(n){
        if(n>0){
            cudaMalloc(&ptr_,n*sizeof(T));
        }
    }

    ~DeviceBuffer(){
        if(ptr_){
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept :ptr_(other.ptr_),size_(other.size_){
        other.ptr_=nullptr;
        other.size_=0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if(this!=&other){
            if(ptr_) cudaFree(ptr_);
            ptr_=other.ptr_;
            size_=other.size_;
            other.ptr_=nullptr;
            other.size_=0;
        }
        return *this;
    }

    void to_device(const T* h_src){
        cudaMemcpy(ptr_,h_src,size_*sizeof(T),cudaMemcpyHostToDevice);
    }

    void to_host(T* h_dst) const {
        cudaMemcpy(h_dst,ptr_,size_*sizeof(T),cudaMemcpyDeviceToHost);
    }

    T* data(){return ptr_;}
    const T* data() const {return ptr_;}
    size_t size() const {return size_;}

    // .view() -> Writeable View | .cview() -> Read-Only Constant View!
    TensorView2D<T> view(int r,int c){return TensorView2D<T>{ptr_,r,c};}
    TensorView2D<const T> cview(int r,int c) const {return TensorView2D<const T>{ptr_,r,c};}
};
