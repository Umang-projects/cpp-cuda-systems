#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_buffer.hpp"

#define TILE 32

// 0. V0: Normal Naive Global Memory (1 Thread = 1 Output Cell, 0 Shared Memory)
__global__ void matmul_v0(
    TensorView2D<const float> A,
    TensorView2D<const float> B,
    TensorView2D<float> C,
    int N
){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;

    if(row<N && col<N){
        float sum=0.0f;
        for(int k=0;k<N;k++){
            sum+=A(row,k)*B(k,col);
        }
        C(row,col)=sum;
    }
}

// 1. V1: Shared Memory Tiled (1 Thread = 1 Output Cell, 32x32 = 1024 threads/block)
__global__ void matmul_v1_tiled(
    TensorView2D<const float> A,
    TensorView2D<const float> B,
    TensorView2D<float> C,
    int N
){
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int col=blockIdx.x*TILE+tx;
    int row=blockIdx.y*TILE+ty;

    float sum=0.0f;
    int num_tiles=N/TILE;

    for(int t=0;t<num_tiles;t++){
        sA[ty][tx]=A(row,t*TILE+tx);
        sB[ty][tx]=B(t*TILE+ty,col);

        __syncthreads();

        for(int k=0;k<TILE;k++){
            sum+=sA[ty][k]*sB[k][tx];
        }

        __syncthreads();
    }

    C(row,col)=sum;
}

// 2. V2: Shared Memory + 2x2 Register Tile (1 Thread = 4 Output Cells, 16x16 = 256 threads/block)
__global__ void matmul_v2_reg_tiled(
    TensorView2D<const float> A,
    TensorView2D<const float> B,
    TensorView2D<float> C,
    int N
){
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int r0=ty*2;
    int c0=tx*2;

    int row=blockIdx.y*TILE+r0;
    int col=blockIdx.x*TILE+c0;

    float sum00=0.0f,sum01=0.0f;
    float sum10=0.0f,sum11=0.0f;

    int num_tiles=N/TILE;

    for(int t=0;t<num_tiles;t++){
        int a_col=t*TILE+c0;
        int b_row=t*TILE+r0;

        sA[r0][c0]=A(row,a_col);
        sA[r0][c0+1]=A(row,a_col+1);
        sA[r0+1][c0]=A(row+1,a_col);
        sA[r0+1][c0+1]=A(row+1,a_col+1);

        sB[r0][c0]=B(b_row,col);
        sB[r0][c0+1]=B(b_row,col+1);
        sB[r0+1][c0]=B(b_row+1,col);
        sB[r0+1][c0+1]=B(b_row+1,col+1);

        __syncthreads();

        for(int k=0;k<TILE;k++){
            float a0=sA[r0][k];
            float a1=sA[r0+1][k];
            float b0=sB[k][c0];
            float b1=sB[k][c0+1];

            sum00+=a0*b0;
            sum01+=a0*b1;
            sum10+=a1*b0;
            sum11+=a1*b1;
        }

        __syncthreads();
    }

    C(row,col)=sum00;
    C(row,col+1)=sum01;
    C(row+1,col)=sum10;
    C(row+1,col+1)=sum11;
}

int main(){
    int N=1024;
    size_t total=(size_t)N*N;
    std::vector<float> h_A(total,1.0f),h_B(total,2.0f);
    std::vector<float> h_C0(total,0.0f),h_C1(total,0.0f),h_C2(total,0.0f);

    DeviceBuffer<float> d_A(total),d_B(total),d_C(total);
    d_A.to_device(h_A.data());
    d_B.to_device(h_B.data());

    int iters=50;
    double total_flops=2.0*(double)N*(double)N*(double)N;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 0. BENCHMARK V0 (Naive Global 16x16: 256 threads/block) ---
    dim3 threads_v0(16,16);
    dim3 blocks_v0(N/16,N/16);
    int blocks_sm_v0=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm_v0,matmul_v0,256,0);

    for(int i=0;i<100;i++){
        matmul_v0<<<blocks_v0,threads_v0>>>(d_A.cview(N,N),d_B.cview(N,N),d_C.view(N,N),N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0;i<iters;i++){
        matmul_v0<<<blocks_v0,threads_v0>>>(d_A.cview(N,N),d_B.cview(N,N),d_C.view(N,N),N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_v0=0.0f;
    cudaEventElapsedTime(&ms_v0,start,stop);
    float avg_v0=ms_v0/iters;
    d_C.to_host(h_C0.data());

    // --- 1. BENCHMARK V1 (Shared 32x32: 1024 threads/block) ---
    dim3 threads_v1(32,32);
    dim3 blocks_v1(N/32,N/32);
    int blocks_sm_v1=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm_v1,matmul_v1_tiled,1024,0);

    for(int i=0;i<100;i++){
        matmul_v1_tiled<<<blocks_v1,threads_v1>>>(d_A.cview(N,N),d_B.cview(N,N),d_C.view(N,N),N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0;i<iters;i++){
        matmul_v1_tiled<<<blocks_v1,threads_v1>>>(d_A.cview(N,N),d_B.cview(N,N),d_C.view(N,N),N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_v1=0.0f;
    cudaEventElapsedTime(&ms_v1,start,stop);
    float avg_v1=ms_v1/iters;
    d_C.to_host(h_C1.data());

    // --- 2. BENCHMARK V2 (2x2 Register Tiled: 256 threads/block) ---
    dim3 threads_v2(16,16);
    dim3 blocks_v2(N/32,N/32);
    int blocks_sm_v2=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm_v2,matmul_v2_reg_tiled,256,0);

    for(int i=0;i<100;i++){
        matmul_v2_reg_tiled<<<blocks_v2,threads_v2>>>(d_A.cview(N,N),d_B.cview(N,N),d_C.view(N,N),N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0;i<iters;i++){
        matmul_v2_reg_tiled<<<blocks_v2,threads_v2>>>(d_A.cview(N,N),d_B.cview(N,N),d_C.view(N,N),N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_v2=0.0f;
    cudaEventElapsedTime(&ms_v2,start,stop);
    float avg_v2=ms_v2/iters;
    d_C.to_host(h_C2.data());

    std::cout<<"Verify C[last] -> V0: "<<h_C0[total-1]
             <<" | V1: "<<h_C1[total-1]
             <<" | V2: "<<h_C2[total-1]<<" (Expected: 2048)\n\n";
    std::cout<<"[V0 Naive Global ( 256 thr/blk)] Blocks/SM: "<<blocks_sm_v0
             <<" | Thr/SM: "<<blocks_sm_v0*256
             <<" | Time: "<<avg_v0<<" ms | Speed: "<<total_flops/(avg_v0*1e6)<<" GFLOPS\n";
    std::cout<<"[V1 Shared 32x32 (1024 thr/blk)] Blocks/SM: "<<blocks_sm_v1
             <<" | Thr/SM: "<<blocks_sm_v1*1024
             <<" | Time: "<<avg_v1<<" ms | Speed: "<<total_flops/(avg_v1*1e6)<<" GFLOPS\n";
    std::cout<<"[V2 RegTile 2x2  ( 256 thr/blk)] Blocks/SM: "<<blocks_sm_v2
             <<" | Thr/SM: "<<blocks_sm_v2*256
             <<" | Time: "<<avg_v2<<" ms | Speed: "<<total_flops/(avg_v2*1e6)<<" GFLOPS\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
