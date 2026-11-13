#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "device_buffer.hpp"

__device__ inline float silu(float x){
    return x/(1.0f+expf(-x));
}

// 1. Scalar Kernel using C++ TensorView2D (Also handles d % 4 != 0 tail sizes safely!)
__global__ void swiglu_v0_view_kernel(TensorView2D<const float> in_v,TensorView2D<float> out_v,int d){
    int row=blockIdx.x;
    for(int j=threadIdx.x;j<d;j+=blockDim.x){
        float x=in_v(row,j);
        float y=in_v(row,d+j);
        out_v(row,j)=silu(x)*y;
    }
}

// 2. Vectorized 128-bit float4 Kernel (Fast path when d % 4 == 0)
__global__ void swiglu_v2_float4_kernel(const float* __restrict__ in,float* __restrict__ out,int d){
    const float4* in4=reinterpret_cast<const float4*>(in);
    float4* out4=reinterpret_cast<float4*>(out);
    int vec_d=d/4;

    int row=blockIdx.x;
    int in_row_start=row*(2*vec_d);
    int out_row_start=row*vec_d;

    for(int j=threadIdx.x;j<vec_d;j+=blockDim.x){
        float4 gx=in4[in_row_start+j];
        float4 gy=in4[in_row_start+vec_d+j];

        float4 res;
        res.x=silu(gx.x)*gy.x;
        res.y=silu(gx.y)*gy.y;
        res.z=silu(gx.z)*gy.z;
        res.w=silu(gx.w)*gy.w;

        out4[out_row_start+j]=res;
    }
}

// 3. Production Dispatcher: Automatically picks float4 or fallback view kernel
void launch_swiglu(const DeviceBuffer<float>& d_in,DeviceBuffer<float>& d_out,int rows,int d){
    int blocks=rows;
    int threads=256;
    if(d%4==0){
        swiglu_v2_float4_kernel<<<blocks,threads>>>(d_in.data(),d_out.data(),d);
    }else{
        swiglu_v0_view_kernel<<<blocks,threads>>>(d_in.cview(rows,2*d),d_out.view(rows,d),d);
    }
}

int main(){
    // TEST A: Standard Benchmark on d = 4096 (Multiple of 4)
    {
        int rows=4096,d=4096;
        size_t in_size=(size_t)rows*(2*d);
        size_t out_size=(size_t)rows*d;

        std::vector<float> h_in(in_size);
        std::vector<float> h_out(out_size,0.0f);
        for(size_t i=0;i<in_size;i++){
            h_in[i]=((int)(i%100)-50)*0.01f;
        }

        DeviceBuffer<float> d_in(in_size);
        DeviceBuffer<float> d_out(out_size);
        d_in.to_device(h_in.data());

        int blocks=rows,threads=256,iters=100;
        double total_bytes=(in_size+out_size)*sizeof(float);

        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // V0 TensorView2D Benchmark (using .cview for input!)
        for(int i=0;i<10;i++){
            swiglu_v0_view_kernel<<<blocks,threads>>>(d_in.cview(rows,2*d),d_out.view(rows,d),d);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int i=0;i<iters;i++){
            swiglu_v0_view_kernel<<<blocks,threads>>>(d_in.cview(rows,2*d),d_out.view(rows,d),d);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_v0=0.0f;
        cudaEventElapsedTime(&ms_v0,start,stop);
        float avg_v0=ms_v0/iters;

        // V2 float4 Benchmark via Dispatcher
        for(int i=0;i<10;i++){
            launch_swiglu(d_in,d_out,rows,d);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int i=0;i<iters;i++){
            launch_swiglu(d_in,d_out,rows,d);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_v2=0.0f;
        cudaEventElapsedTime(&ms_v2,start,stop);
        float avg_v2=ms_v2/iters;

        std::cout<<"=== TEST A: d=4096 (Aligned float4 vs TensorView2D Scalar) ===\n";
        std::cout<<"[V0 TensorView2D] Avg Time: "<<avg_v0<<" ms | Bandwidth: "<<total_bytes/(avg_v0*1e6)<<" GB/s\n";
        std::cout<<"[V2 float4      ] Avg Time: "<<avg_v2<<" ms | Bandwidth: "<<total_bytes/(avg_v2*1e6)<<" GB/s\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // TEST B: Odd Tail Size d = 4098 (Not divisible by 4 -> Dispatcher uses Safe Fallback!)
    {
        int rows=16,d=4098;
        size_t in_size=(size_t)rows*(2*d);
        size_t out_size=(size_t)rows*d;
        std::vector<float> h_in(in_size,1.0f);
        std::vector<float> h_out(out_size,0.0f);

        DeviceBuffer<float> d_in(in_size);
        DeviceBuffer<float> d_out(out_size);
        d_in.to_device(h_in.data());

        launch_swiglu(d_in,d_out,rows,d);
        cudaDeviceSynchronize();
        d_out.to_host(h_out.data());

        std::cout<<"\n=== TEST B: d=4098 (Tail Size Fallback Check) ===\n";
        std::cout<<"Last element out[rows*d-1]: "<<h_out[out_size-1]<<" (Expected ~0.731059)\n";
    }

    return 0;
}
