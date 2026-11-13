# ⚡ C++ Systems & GPU Kernel Engineering From First Principles

Exploring C++ systems programming (RAII, Rule of 5, zero-cost non-owning views) and hardware-aware CUDA kernel execution (SM warp scheduling, register pressure, cache-line coalescing, and vectorized memory transactions) on NVIDIA GPUs.

---

## 🛠️ Kernel #1: Fused `SwiGLU` (`SiluAndMul`) — From Naive Baseline to 128-Bit Vectorized `float4`

### 1. Mathematical & Memory Layout Blueprint

Modern LLMs (Llama-3, Qwen, DeepSeek) concatenate the **Gate ($x$)** and **Up/Signal ($y$)** projections into a single contiguous tensor of shape `[rows, 2 * d]`:

$$\text{SwiGLU}(x, y) = \text{SiLU}(x) \times y = \left(\frac{x}{1 + e^{-x}}\right) \times y$$

```text
INPUT TENSOR `in` [rows, 2*d] (1D Contiguous Row-Major Layout in VRAM):
=======================================================================
              <------ GATE (x): 0 .. d-1 ------>│<---- SIGNAL (y): d .. 2d-1 ---->
            ┌────────┬────────┬────────┬────────┐┌────────┬────────┬────────┬────────┐
Row 0       │  x0,0  │  x0,1  │  ...   │ x0,d-1 ││  y0,0  │  y0,1  │  ...   │ y0,d-1 │ <-- Block 0 (blockIdx.x = 0)
            ├────────┼────────┼────────┼────────┤├────────┼────────┼────────┼────────┤
Row 1       │  x1,0  │  x1,1  │  ...   │ x1,d-1 ││  y1,0  │  y1,1  │  ...   │ y1,d-1 │ <-- Block 1 (blockIdx.x = 1)
            └────────┴────────┴────────┴────────┘└────────┴────────┴────────┴────────┘
                 │                                    │
                 └──────────────► ( × ) ◄─────────────┘  out(row, j) = SiLU(in(row, j)) * in(row, d + j)
                                    │
                                    ▼
OUTPUT TENSOR `out` [rows, d]:  [ out(row, 0) ... out(row, d-1) ] (100% Independent across pairs -> Zero Race Conditions)
```

---

### 2. C++ Systems Architecture (`device_buffer.hpp`)

Instead of raw `cudaMalloc` / `cudaFree` pointers in host code, memory ownership and kernel indexing are decoupled into two layers:

```text
HOST (CPU Stack — Ownership Layer)                    DEVICE (GPU Constant Memory & Registers — View Layer)
==================================                    =====================================================
DeviceBuffer<float> d_in(rows * 2 * d);               TensorView2D<const float> in_v = d_in.cview(rows, 2*d);
┌────────────────────────────────────┐                ┌────────────────────────────────────────────────┐
│ • Owns VRAM lifecycle (RAII)       │                │ • ptr_  = 0x7f8a00000000 (8B)                  │
│ • Copy Constructor = DELETED       │ ──(.cview())─► │ • rows_ = 4096 (4B), cols_ = 8192 (4B)         │
│ • Move Constructor = Steals & Nulls│                │ • Total = 16 Bytes (Trivially Copyable to GPU) │
│ • ~DeviceBuffer()  = Auto cudaFree │                │ • inline operator()(r, c) -> ptr_[r*cols_ + c] │
└────────────────────────────────────┘                └────────────────────────────────────────────────┘
```

#### **Why Decouple `DeviceBuffer<T>` and `TensorView2D<T>`?**
1. **Rule of 5 vs CUDA Pass-by-Value:** `DeviceBuffer<T>` deletes its copy constructor to prevent double-free bugs. CUDA kernels pass arguments by value into Constant Memory (`cmem[0]`), so passing `DeviceBuffer<T>` directly is illegal and would trigger device-side destructors.
2. **Hardware Constant Memory Verification (`ptxas`):**
   * Passing raw pointers `(const float*, float*, int)` consumes **`372 bytes` of `cmem[0]`**.
   * Passing two `TensorView2D` structs `(in_v, out_v, int)` consumes **`388 bytes` of `cmem[0]`** (`+16 bytes` for two `{rows_, cols_}` pairs) with **zero runtime clock-cycle overhead** (`inline` zero-cost abstraction).
3. **GPU Memory Leak Verification (`compute-sanitizer --tool memcheck --leak-check full`):**
   * **Raw Pointers (Without RAII):** `LEAK SUMMARY: 96 bytes leaked in 2 allocations` (`64B` input + `32B` output at 512B-aligned driver addresses via `cuMemAlloc_v2`).
   * **With `DeviceBuffer<T>` RAII:** `LEAK SUMMARY: 0 bytes leaked in 0 allocations | ERROR SUMMARY: 0 errors`.

---

### 3. GPU Hardware Execution & Memory Hierarchy (`Scalar float` vs `Vectorized float4`)

Each row (token) is assigned to **1 Thread Block (`row = blockIdx.x`)** with **`blockDim.x = 256` threads (`8 Warps`)** executing a block-stride loop (`j += blockDim.x`).

```text
==========================================================================================
SCALAR KERNEL (V0: `float` = 4 Bytes per Thread Load -> `LD.E.32`)
==========================================================================================
• 1 Warp (32 Threads) × 4 Bytes = 128 Bytes per instruction (1 L1 Cache Line)
• For d = 4096: 256 threads require 16 Stride Rounds per row
• Register Usage (ptxas sm_75): 18 Registers / Thread

  [ VRAM ] ──(128B: 1 Cache Line)──► [ L2 ] ──► [ L1 ] ──► [ 1 Reg (x) + 1 Reg (y) per Thread ]


==========================================================================================
VECTORIZED KERNEL (V2: `float4` = 16 Bytes per Thread Load -> `LD.E.128`)
==========================================================================================
• 1 Warp (32 Threads) × 16 Bytes = 512 Bytes per instruction (4 L1 Cache Lines in 1 burst!)
• For d = 4096 (vec_d = 1024 crates): 256 threads require ONLY 4 Stride Rounds per row (4x fewer loops!)
• Register Usage (ptxas sm_75): 24 Registers / Thread (+6 Regs to hold .x, .y, .z, .w in flight)

  [ VRAM ] ──(512B: 4 Cache Lines!)─► [ L2 ] ──► [ L1 ] ──► [ 4 Regs (gx) + 4 Regs (gy) per Thread ]
```

#### **Hardware Trade-Offs Observed (`float` vs `float4`):**
| Hardware Metric | `V0` Scalar (`float` + `TensorView2D`) | `V2` Vectorized (`float4`) | Hardware Mechanism & Trade-Off |
| :--- | :--- | :--- | :--- |
| **Load Instruction Width** | `32-bit` (`4 Bytes`) | `128-bit` (`16 Bytes`) | `4x` more bytes in flight per instruction (Higher Memory-Level Parallelism). |
| **Stride Loop Iterations (`d=4096`)** | `16` rounds | `4` rounds | `4x` fewer loop branches & address arithmetic instructions on Warp Schedulers. |
| **Registers / Thread (`ptxas`)** | `18` registers | `24` registers | `float4` consumes more 32-bit registers (`+6 regs`). On T4 (`65,536` regs/SM, max `1024` threads/SM), `1024 × 24 = 24,576 < 65,536`, maintaining **100% Occupancy** with `0 bytes` register spilling. |
| **Constant Memory (`cmem[0]`)** | `388 bytes` | `372 bytes` | `+16 bytes` in V0 due to `rows_` and `cols_` inside two `TensorView2D` arguments. |
| **16-Byte Alignment Requirement** | Works on any `d` | Requires `d % 4 == 0` | If `d = 4098`, the Signal half starts at byte offset `4098 × 4 = 16,392` (`16,392 % 16 = 8 != 0`), which would fault on `LD.E.128`. Solved via host dispatcher (`launch_swiglu`). |

---

### 4. Key Engineering Bugs Caught & Fixed During Development

1. **C++ `size_t` Unsigned Arithmetic Underflow:**
   * Writing `((i % 100) - 50) * 0.01f` with `size_t i = 0` caused `0UL - 50` to wrap around to `18,446,744,073,709,551,566` ($1.84 \times 10^{19}$), producing `out[0] = 8.4855e+16` instead of `-0.0868343`. Fixed via explicit signed cast `((int)(i % 100) - 50)`.
2. **Template Const-Qualification (`TensorView2D<float>` vs `TensorView2D<const float>`):**
   * Calling `.view()` on a non-const `DeviceBuffer<float>` returns `TensorView2D<float>`, which cannot implicitly convert to `TensorView2D<const float>`. Solved by adding an explicit `.cview()` method (akin to `std::vector::cbegin()`).
3. **Architecture Target Mismatch (`sm_52` vs `sm_75`):**
   * Without `-arch=sm_75`, `nvcc` defaulted to Maxwell `sm_52` PTX/SASS generation (`212.6 GB/s` on V0). Adding `-arch=sm_75` unlocked native Turing instruction scheduling (`224.9 GB/s` on V0).

---

### 5. Benchmark Results on NVIDIA Tesla T4 (`sm_75`, CUDA 12.4)

* **Workload:** `rows = 4096, d = 4096` (`33.55M` input floats + `16.77M` output floats = **`201.33 MB`** total memory traffic per iteration, averaged over 100 runs after 10 warmup runs).

| Kernel Version | Execution Time (ms) | Effective Bandwidth (GB/s) | Registers / Thread | Register Spills | Tail-Safe (`d=4098`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`V0`: Scalar + `TensorView2D`** | `0.895 ms` | `224.87 GB/s` | `18` | `0 bytes` | ✅ Yes |
| **`V2`: Vectorized `float4` (`LD.E.128`)** | **`0.824 ms`** | **`244.18 GB/s`** | `24` | `0 bytes` | ✅ Via `launch_swiglu` Dispatcher |
