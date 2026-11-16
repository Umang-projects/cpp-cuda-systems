# Dense Matrix Multiplication (`GEMM`: $C = A \times B$) — Complete Hardware Resource, Shared Memory, Bank Conflict & Register Tiling Blueprint

A first-principles hardware and algorithm dissection of dense matrix multiplication (`C = A * B`) on **NVIDIA Tesla T4 (`sm_75`, Turing Architecture, CUDA 12.4)**—progressing from a Naive Global Memory baseline (`V0`) to Shared Memory Tiling (`V1`), Shared Memory Bank-Conflict Padding (`[32][33]`), and 2D Register Tiling (`V2`) achieving **1,875.1 GFLOPS (1.88 TFLOPS)** with **100.0% SM Register File saturation (`65,536 / 65,536` registers)**.

---

## 0. Ground-Truth Numerical Example (`4 x 4` Matrices, `N = 4`)

Every kernel design (`V0`, `V1`, and `V2`) in this document is traced step-by-step against the exact same `4 x 4` matrices below, solving for the **Top-Left `2 x 2` Output Sub-Matrix** (`C(0,0) = 70, C(0,1) = 30, C(1,0) = 39, C(1,1) = 15`):

```text
MATRIX A (4x4):                        MATRIX B (4x4):                        OUTPUT MATRIX C (4x4):
        t = 0       t = 1                     Col 0  Col 1 │ Col 2  Col 3            Col 0   Col 1 │ Col 2  Col 3
     Col0  Col1 │ Col2  Col3            t=0 ┌──────────────┼──────────────┐       ┌────────────────┼─────────────┐
   ┌────────────┼────────────┐          R0  │   5      1   │   0      0   │    R0 │ C(0,0)  C(0,1) │   .      .  │
R0 │   1     2  │   3     4  │          R1  │   6      2   │   0      0   │       │  = 70    = 30  │             │
R1 │   2     1  │   1     2  │              ├──────────────┼──────────────┤    R1 │ C(1,0)  C(1,1) │   .      .  │
   ├────────────┼────────────┤          t=1 │   7      3   │   0      0   │       │  = 39    = 15  │             │
R2 │   0     0  │   0     0  │          R3  │   8      4   │   0      0   │       ├────────────────┼─────────────┤
R3 │   0     0  │   0     0  │              └──────────────┴──────────────┘    R2 │    .       .   │   .      .  │
   └────────────┴────────────┘                                                 R3 │    .       .   │   .      .  │
                                                                                  └────────────────┴─────────────┘
GROUND-TRUTH DOT-PRODUCT MATH:
  • C(0,0) = (1*5 + 2*6) [t=0: 17]  +  (3*7 + 4*8) [t=1: 53]  = 70
  • C(0,1) = (1*1 + 2*2) [t=0:  5]  +  (3*3 + 4*4) [t=1: 25]  = 30
  • C(1,0) = (2*5 + 1*6) [t=0: 16]  +  (1*7 + 2*8) [t=1: 23]  = 39
  • C(1,1) = (2*1 + 1*2) [t=0:  4]  +  (1*3 + 2*4) [t=1: 11]  = 15
```

---
---

## 1. Naive Global Memory Baseline (`V0`: `matmul_v0`)

### 1.1 Side-by-Side Numerical Dry Run & CUDA Hardware Mapping

In `matmul_v0`, each GPU thread computes **1 output cell `C(row, col)`** by reading an entire row of `A` and an entire column of `B` directly from Global Memory (VRAM / L2):

```text
DETAILED DRY RUN (Numbers):                               CUDA CODE + HARDWARE RESOURCE (Side-by-Side):
===========================                               =============================================
1. 2D Thread Coordinates:                           --->  int col=blockIdx.x*blockDim.x+threadIdx.x; // [Reg R0: .x Col]
   T(0,0)->(0,0) | T(0,1)->(0,1)                          int row=blockIdx.y*blockDim.y+threadIdx.y; // [Reg R1: .y Row]
   T(1,0)->(1,0) | T(1,1)->(1,1)

2. Bounds Check & Private Accumulator Register:     --->  if(row<N && col<N){
                                                              float sum=0.0f;                        // [Reg R2: Accum]
3. Sequential Loop k = 0, 1, 2, 3 (VRAM Reads):     --->      for(int k=0;k<N;k++){
   • T(0,0) reads A Row 0 [1,2,3,4] & B Col 0 [5,6,7,8]:          sum+=A(row,k)*B(k,col);            // [VRAM Read x2]
     k=0: 1*5=5 | k=1: +12=17 | k=2: +21=38 | k=3: +32=70     }

   • T(0,1) reads A Row 0 [1,2,3,4] AGAIN & B Col 1 [1,2,3,4]:
     k=0: 1*1=1 | k=1: +4=5   | k=2: +9=14  | k=3: +16=30

   • T(1,0) reads A Row 1 [2,1,1,2] & B Col 0 [5,6,7,8] AGAIN:
     k=0: 2*5=10| k=1: +6=16  | k=2: +7=23  | k=3: +16=39

   • T(1,1) reads A Row 1 [2,1,1,2] AGAIN & B Col 1 [1,2,3,4] AGAIN:
     k=0: 2*1=2 | k=1: +2=4   | k=2: +3=7   | k=3: +8=15

4. Store Final Register `sum` to VRAM Matrix C:     --->      C(row,col)=sum;                        // [VRAM Write]
   C(0,0)=70, C(0,1)=30, C(1,0)=39, C(1,1)=15             }
```

### 1.2 Hardware Resource Bottleneck in `V0` (Global Memory Redundancy)

```text
VRAM CAPACITY vs. REDUNDANT VRAM READ TRAFFIC (For N = 1024 x 1024):
====================================================================
• Actual Size of Matrix A + Matrix B in VRAM : 4 MB + 4 MB = 8 MB
• Shared Memory Used per SM                  : 0 Bytes (Out of 64 KB available per SM on Tesla T4!)
• Reads per Thread (k = 0..1023)             : 1024 floats (A) + 1024 floats (B) = 2,048 floats (8 KB)
• Total Reads Across all 1,048,576 Threads   : 1,048,576 threads * 8 KB = 8.38 GB of Read Requests!
• Bottleneck                                 : Every row of A and column of B is fetched 1,024 separate times!
```

### 1.3 Block Size vs. SM Occupancy & L1 Cache Line Analysis on Tesla T4 (`sm_75`)

* **Tesla T4 Hardware Limits per SM (40 SMs total):**
  * Max Resident Threads per SM: **`1,024` Threads (`32` Warps)**
  * Max Resident Blocks per SM: **`16` Blocks**
  * Total 32-Bit Registers per SM: **`65,536` Registers** (Budget for 100% Occupancy = `65,536 / 1,024 = 64 Registers/Thread`)
  * Shared Memory Capacity per SM: **`64 KB` (`65,536 Bytes`)**
* **`ptxas` Output for `matmul_v0`:** `Used 52 registers, 0 bytes smem, 404 bytes cmem[0]` (`1,024 * 52 = 53,248 <= 65,536` registers/SM $\rightarrow$ `0` spills).

| Block Dimensions (`BS x BS`) | Threads / Block | Active Blocks / SM | Active Threads / SM | Hardware Occupancy | Avg Time (`N=1024`) | Hardware Bottleneck & Warp Coalescing Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`4 x 4`** | `16` (`0.5` Warp) | `16` (Max Limit) | `256 / 1024` | `25.0%` | `12.80 ms` | Hits T4's `16 Blocks/SM` hard ceiling; each 16-thread block wastes half of a 32-thread hardware warp. |
| **`8 x 8`** | `64` (`2` Warps) | `16` (Max Limit) | `1024 / 1024` | `100.0%` | `5.69 ms` | `blockDim.x = 8`: 1 Warp (`32` threads) splits across 4 different rows, using only `32B` of each `128B` L1 cache line of `B`. |
| **`10 x 10`** | `100` (`3.125` Warps) | `8` | `800 / 1024` | `78.125%` | `6.69 ms` | Not a multiple of 32: requires `ceil(100/32) = 4 Warps` (`128` slots) per block, wasting `28` thread slots per block (`8 * 28 = 224` wasted slots/SM). |
| **`16 x 16`** | `256` (`8` Warps) | `4` | `1024 / 1024` | `100.0%` | `4.02 ms` | Balanced `4` blocks/SM concurrency; 16 threads per row coalesce `64B` (half an L1 cache line) of `B`. |
| **`32 x 32`** | `1024` (`32` Warps) | `1` | `1024 / 1024` | `100.0%` | `3.09 ms` | `blockDim.x = 32` equals 1 full Warp: all 32 threads share the same `row` (`A(row, k)` broadcasts in 1 tick) and read `col = 0..31` of `B(k, col)` (1 full `128B` L1 cache line). |

### 1.4 Small Matrix (`N = 128`) vs. Large Matrix (`N = 1024`) Wave Quantization & SM Starvation

Choosing block size also depends on total grid blocks relative to the GPU's `40 SMs`:

```text
CASE A: Small Matrix N = 128 x 128 (Total Output Cells = 16,384):
  • If Block = 32x32 (1,024 thr): Total Blocks = 16,384 / 1,024 = ONLY 16 Blocks!
    -> On 40 SMs, only 16 SMs receive 1 Block; 24 SMs (60% of the GPU!) sit 100% IDLE!
  • If Block = 16x16 (256 thr)  : Total Blocks = 16,384 / 256 = 64 Blocks.
    -> Round 1 distributes 40 Blocks across SM 0..39 (1 Block each).
    -> Round 2 distributes remaining (64 - 40 = 24) Blocks across SM 0..23.
    -> Result: SM 0..23 hold 2 Blocks (512 thr = 50% occ), SM 24..39 hold 1 Block (256 thr = 25% occ).

CASE B: Large Matrix N = 1024 x 1024 (Total Output Cells = 1,048,576):
  • If Block = 32x32 (1,024 thr): Total Blocks = 1,048,576 / 1,024 = 1,024 Blocks.
    -> 1,024 Blocks / 40 SMs = 25.6 Waves:
       First 25 waves run 40 * 25 = 1,000 Blocks across all 40 SMs.
       The remaining 24 Blocks (24 / 40 = 0.6 partial wave) run on 24 SMs in the 26th wave (Tail Effect).
```

---
---

## 2. On-Chip Shared Memory Tiled Redesign (`V1`: `matmul_v1_tiled`)

To eliminate redundant VRAM reads, threads inside a `TILE x TILE` block collaboratively stage `TILE x TILE` chunks of `A` and `B` into on-chip **Shared Memory (`__shared__` SRAM)**, sliding across `num_tiles = N / TILE` phases:

```text
SLIDING TILE EXECUTION PIPELINE (Tile t = 0 .. N/TILE - 1):
=====================================================================================
  Matrix A [row, t*TILE + tx] ──► __shared__ float sA[TILE][TILE] ┐
                                                                  ├──► __syncthreads() (#1: Wait for Tile Load)
  Matrix B [t*TILE + ty, col] ──► __shared__ float sB[TILE][TILE] ┘
                                                                  │
                                                                  ▼
                                     Compute Dot-Product from SRAM (k = 0 .. TILE-1):
                                       sum += sA[ty][k] * sB[k][tx]   (Reused TILE times!)
                                                                  │
                                                                  ▼
                                                           __syncthreads() (#2: Wait before next Tile overwrite)
=====================================================================================
```

---

### 2.1 Detailed Block-by-Block Numerical Dry Run & CUDA Mapping (`V1`, `TILE = 2` on `4x4` Matrix)

#### **DESIGN BLOCK 1: Setup (Shared Tables + 4 Threads in `Block(0,0)`)**

```text
DETAILED DRY RUN (Numbers & Coordinates):                 CUDA CODE + HARDWARE RESOURCE (Side-by-Side):
=========================================                 =============================================
1. Allocate two 2x2 Shared Memory Tables in SRAM:   --->  __shared__ float sA[2][2]; // [SRAM: 16B (2KB for 16x16)]
   sA = [ ?  ? ]      sB = [ ?  ? ]                       __shared__ float sB[2][2]; // [SRAM: 16B (2KB for 16x16)]
        [ ?  ? ]           [ ?  ? ]

2. Local & Global Coordinates for 4 Threads:        --->  int tx=threadIdx.x;        // [Reg: 0 or 1]
   • T(0,0): ty=0, tx=0 -> row = 0*2+0 = 0, col = 0       int ty=threadIdx.y;        // [Reg: 0 or 1]
   • T(0,1): ty=0, tx=1 -> row = 0*2+0 = 0, col = 1       int row=blockIdx.y*2+ty;   // [Reg: Global Row]
   • T(1,0): ty=1, tx=0 -> row = 0*2+1 = 1, col = 0       int col=blockIdx.x*2+tx;   // [Reg: Global Col]
   • T(1,1): ty=1, tx=1 -> row = 0*2+1 = 1, col = 1

3. Private Accumulator & Total Tiles (4/2 = 2):     --->  float sum=0.0f;            // [Reg: Accumulator]
   T(0,0)..T(1,1): sum = 0.0f                             int num_tiles=N/2;         // [Reg: 2 Tiles]
```

#### **DESIGN BLOCK 2: Outer Loop Round 1 (`t = 0` — First `2 x 2` Tile)**

```text
DETAILED DRY RUN FOR t = 0:                               CUDA CODE + HARDWARE RESOURCE (Side-by-Side):
===========================                               =============================================
                                                          for(int t=0;t<num_tiles;t++){ // [t = 0]

STEP 2.1: Calculate Col of A & Row of B for t = 0:
  Formula: a_col = t*2 + tx  |  b_row = t*2 + ty    --->      int a_col=t*2+tx;         // (0*2 + tx = 0, 1)
  • T(0,0) [ty=0,tx=0]: a_col = 0, b_row = 0                  int b_row=t*2+ty;         // (0*2 + ty = 0, 1)
  • T(0,1) [ty=0,tx=1]: a_col = 1, b_row = 0
  • T(1,0) [ty=1,tx=0]: a_col = 0, b_row = 1
  • T(1,1) [ty=1,tx=1]: a_col = 1, b_row = 1
-------------------------------------------------------------------------------------------------------
STEP 2.2: Collaborative Load (1 float/thread):      --->      sA[ty][tx]=A(row,a_col);  // [VRAM -> SRAM]
  • T(0,0): sA[0][0] = A(0,0) = 1 | sB[0][0] = B(0,0) = 5     sB[ty][tx]=B(b_row,col);  // [VRAM -> SRAM]
  • T(0,1): sA[0][1] = A(0,1) = 2 | sB[0][1] = B(0,1) = 1
  • T(1,0): sA[1][0] = A(1,0) = 2 | sB[1][0] = B(1,0) = 6
  • T(1,1): sA[1][1] = A(1,1) = 1 | sB[1][1] = B(1,1) = 2

  Shared Tables in SRAM now hold:
       Col0 Col1             Col0 Col1
  sA = [ 1   2 ]        sB = [ 5   1 ]
       [ 2   1 ]             [ 6   2 ]
-------------------------------------------------------------------------------------------------------
STEP 2.3: Barrier #1 (Wait until SRAM is filled):   --->      __syncthreads();          // [RAW Hazard Guard]
-------------------------------------------------------------------------------------------------------
STEP 2.4: Inner Loop (k = 0, then k = 1) from SRAM: --->      for(int k=0;k<2;k++){
                                                                  sum+=sA[ty][k]*sB[k][tx];
  [Sub-step k = 0]:                                           }
  • T(0,0): sum += sA[0][0]*sB[0][0] = 1*5 = 5
  • T(0,1): sum += sA[0][0]*sB[0][1] = 1*1 = 1
  • T(1,0): sum += sA[1][0]*sB[0][0] = 2*5 = 10
  • T(1,1): sum += sA[1][0]*sB[0][1] = 2*1 = 2

  [Sub-step k = 1]:
  • T(0,0): sum = 5  + sA[0][1]*sB[1][0] = 5  + (2*6) = 17
  • T(0,1): sum = 1  + sA[0][1]*sB[1][1] = 1  + (2*2) = 5
  • T(1,0): sum = 10 + sA[1][1]*sB[1][0] = 10 + (1*6) = 16
  • T(1,1): sum = 2  + sA[1][1]*sB[1][1] = 2  + (1*2) = 4
-------------------------------------------------------------------------------------------------------
STEP 2.5: Barrier #2 (Wait before t=1 overwrite):   --->      __syncthreads();          // [WAR Hazard Guard]
```

#### **DESIGN BLOCK 3: Outer Loop Round 2 (`t = 1` — Second `2 x 2` Tile)**

```text
DETAILED DRY RUN FOR t = 1:                               CUDA CODE + HARDWARE RESOURCE (Side-by-Side):
===========================                               =============================================
                                                          // Next iteration of for(int t...) [t = 1]

STEP 3.1: Calculate new Col of A & Row of B (Jump +2):
  Formula: a_col = 1*2 + tx  |  b_row = 1*2 + ty    --->      int a_col=t*2+tx;         // (1*2 + tx = 2, 3)
  • T(0,0) [ty=0,tx=0]: a_col = 2, b_row = 2                  int b_row=t*2+ty;         // (1*2 + ty = 2, 3)
  • T(0,1) [ty=0,tx=1]: a_col = 3, b_row = 2
  • T(1,0) [ty=1,tx=0]: a_col = 2, b_row = 3
  • T(1,1) [ty=1,tx=1]: a_col = 3, b_row = 3
-------------------------------------------------------------------------------------------------------
STEP 3.2: Overwrite SAME sA & sB with t=1 numbers:  --->      sA[ty][tx]=A(row,a_col);  // [Overwrite SRAM]
  • T(0,0): sA[0][0] = A(0,2) = 3 | sB[0][0] = B(2,0) = 7     sB[ty][tx]=B(b_row,col);  // [Overwrite SRAM]
  • T(0,1): sA[0][1] = A(0,3) = 4 | sB[0][1] = B(2,1) = 3
  • T(1,0): sA[1][0] = A(1,2) = 1 | sB[1][0] = B(3,0) = 8
  • T(1,1): sA[1][1] = A(1,3) = 2 | sB[1][1] = B(3,1) = 4

  Shared Tables in SRAM now hold:
       Col0 Col1             Col0 Col1
  sA = [ 3   4 ]        sB = [ 7   3 ]
       [ 1   2 ]             [ 8   4 ]
-------------------------------------------------------------------------------------------------------
STEP 3.3: Barrier #1 (Wait until SRAM is updated):  --->      __syncthreads();          // [RAW Hazard Guard]
-------------------------------------------------------------------------------------------------------
STEP 3.4: Inner Loop (k = 0, then k = 1) adds to sum:-->      for(int k=0;k<2;k++){
                                                                  sum+=sA[ty][k]*sB[k][tx];
  [Sub-step k = 0]:                                           }
  • T(0,0): sum = 17 + sA[0][0]*sB[0][0] = 17 + (3*7) = 38
  • T(0,1): sum = 5  + sA[0][0]*sB[0][1] = 5  + (3*3) = 14
  • T(1,0): sum = 16 + sA[1][0]*sB[0][0] = 16 + (1*7) = 23
  • T(1,1): sum = 4  + sA[1][0]*sB[0][1] = 4  + (1*3) = 7

  [Sub-step k = 1]:
  • T(0,0): sum = 38 + sA[0][1]*sB[1][0] = 38 + (4*8) = 70  (Matches Ground Truth 70!)
  • T(0,1): sum = 14 + sA[0][1]*sB[1][1] = 14 + (4*4) = 30  (Matches Ground Truth 30!)
  • T(1,0): sum = 23 + sA[1][1]*sB[1][0] = 23 + (2*8) = 39  (Matches Ground Truth 39!)
  • T(1,1): sum = 7  + sA[1][1]*sB[1][1] = 7  + (2*4) = 15  (Matches Ground Truth 15!)
-------------------------------------------------------------------------------------------------------
STEP 3.5: Barrier #2 & Final Write to VRAM `C`:     --->      __syncthreads();
                                                          }
                                                          if(row<N && col<N) C(row,col)=sum;
```

---

### 2.2 Critical Engineering Findings in `V1`

#### **Finding #1: Why Two `__syncthreads()` Barriers Are Mandatory Inside the Loop**
1. **First `__syncthreads()` (After loading `sA` and `sB`):** Prevents **Read-After-Write (RAW) hazards**—without it, fast threads would read `sA[ty][k]` before neighboring threads finished writing their element from VRAM.
2. **Second `__syncthreads()` (After the `k` compute loop):** Prevents **Write-After-Read (WAR) hazards**—without it, fast threads would advance to `t + 1` and overwrite `sA` and `sB` while slower warps in the same block were still reading `sA` and `sB` for tile `t`.

#### **Finding #2: The Boundary-Check Tax & Compile-Time Specialization (`<TILE, CHECK_BOUNDS>`)**
* For matrix sizes not divisible by `TILE` (e.g., `N = 1000`), out-of-bounds threads cannot `return` early without causing a `__syncthreads()` deadlock; instead, they pad `sA` and `sB` with `0.0f`:
  ```cpp
  sA[ty][tx] = (row < N && a_col < N) ? A(row, a_col) : 0.0f;
  ```
* **Observed Performance Penalty:** Evaluating runtime bounds predicates inside the hot `t` loop slowed `N = 1024` down by **~25–40%** (`681 GFLOPS` $\rightarrow$ `394 GFLOPS`).
* **Fix:** Specialize the kernel with `template<int TILE, bool CHECK_BOUNDS>`. When `N % TILE == 0`, the host dispatcher launches `CHECK_BOUNDS = false` (zero in-loop branches); when `N % TILE != 0` (`N = 1000`), it dispatches `CHECK_BOUNDS = true`.

#### **Finding #3: Dynamic Boost-Clock Warmup (`5` vs `100` Warmups on `70W` Tesla T4)**
* With only `5` warmup runs (`~15 ms`), the Tesla T4 remained near its base clock (`~850 MHz`), yielding `502–582 GFLOPS`.
* Running `100` warmup iterations (`~250 ms` sustained load) ramped the GPU clock to its `1590 MHz` Boost Clock, increasing `V1` throughput from `582 GFLOPS` to **`1,059.5 GFLOPS` (`1.06 TFLOPS`)**.

---
---

## 3. Shared Memory 32-Bank Architecture & Bank Conflict Padding (`[32][32]` vs `[32][33]`)

On NVIDIA GPUs, Shared Memory is partitioned into **32 independent 32-bit (`4-byte`) Banks (`Bank 0 .. Bank 31`)** that can service 32 simultaneous requests per clock tick **only if the requests target different banks** (or broadcast the exact same address):

$$\text{Bank ID} = \left(\text{1D Float Index}\right) \pmod{32}$$

```text
====================================================================================================
CASE 1: __shared__ float tile[32][32] (Row Stride = 32) — Column-Wise Access `tile[tx][0]`
====================================================================================================
When 32 threads in a Warp (tx = 0..31) access down Column 0 across different rows:
  • Thread tx = 0  ---> tile[0][0]  ---> 1D Index =  0 * 32 + 0 =   0  --->   0 % 32 = Bank 0
  • Thread tx = 1  ---> tile[1][0]  ---> 1D Index =  1 * 32 + 0 =  32  --->  32 % 32 = Bank 0  (CONFLICT!)
  • Thread tx = 2  ---> tile[2][0]  ---> 1D Index =  2 * 32 + 0 =  64  --->  64 % 32 = Bank 0  (CONFLICT!)
  ...
  • Thread tx = 31 ---> tile[31][0] ---> 1D Index = 31 * 32 + 0 = 992  ---> 992 % 32 = Bank 0  (CONFLICT!)

RESULT: All 32 threads collide on Bank 0 (while Banks 1..31 sit idle) -> 32-Way Bank Conflict (32x slower!).


====================================================================================================
CASE 2: __shared__ float tile[32][33] (Row Stride = 33, +1 Dummy Column Padding) — `tile[tx][0]`
====================================================================================================
Adding +1 dummy float per row shifts every subsequent row by +1 Bank:
  • Thread tx = 0  ---> tile[0][0]  ---> 1D Index =  0 * 33 + 0 =    0 --->    0 % 32 = Bank 0
  • Thread tx = 1  ---> tile[1][0]  ---> 1D Index =  1 * 33 + 0 =   33 --->   33 % 32 = Bank 1  (SHIFTED!)
  • Thread tx = 2  ---> tile[2][0]  ---> 1D Index =  2 * 33 + 0 =   66 --->   66 % 32 = Bank 2  (SHIFTED!)
  ...
  • Thread tx = 31 ---> tile[31][0] ---> 1D Index = 31 * 33 + 0 = 1023 ---> 1023 % 32 = Bank 31 (SHIFTED!)

RESULT: All 32 threads map to 32 distinct Banks (Bank 0..31) -> ZERO Bank Conflict (1 clock tick)!
Cost  : Only 32 * 1 * 4B = 128 extra bytes of Shared Memory per tile.
====================================================================================================
```

*(Note: In our standard `V1` inner loop `sum += sA[ty][k] * sB[k][tx]`, threads in a warp have identical `ty` and `k` and varying `tx = 0..31`. Thus `sA[ty][k]` is a single-address **Hardware Broadcast** and `sB[k][tx]` accesses 32 consecutive columns across `Bank 0..31`, making it naturally bank-conflict-free).*

---
---

## 4. 2D Register Tiling / Thread Coarsening (`V2`: `matmul_v2_reg_tiled`)

### 4.1 Why `V1` Still Has a Hardware Bottleneck

In `V1` (`TILE = 32`), two hardware bottlenecks remain:
1. **High Shared-Memory-to-FMA Ratio (`2 : 1`):** To compute 4 output cells (`2 x 2`), 4 separate threads issue **`4 * 2 = 8` Shared Memory loads** per `k` step with zero register reuse.
2. **Single-Block SM Stall (`Blocks/SM = 1`):** A `32 x 32` block requires `1,024` threads, occupying the entire SM thread capacity (`1,024 / 1,024`). Whenever the block hits `__syncthreads()`, all 4 warp schedulers on the SM stall.

### 4.2 The `2 x 2` Register Tile Solution (`1 Thread = 4 Output Cells in Registers`)

We keep the Shared Memory tile at `32 x 32` (`1,024` cells), but reduce the thread block to **`16 x 16 = 256` threads**. Each single thread `(ty, tx)` computes a **`2 x 2` patch (`4` cells: `sum00, sum01, sum10, sum11`)** in its private registers:

```text
32x32 TILE KE ANDAR 1 THREAD (ty, tx) KA 2x2 REGISTER TILE:
===========================================================
Local Corner: r0 = ty * 2,  c0 = tx * 2

                     b0 = sB[k][c0]     b1 = sB[k][c0+1]
                   ┌──────────────────┬──────────────────┐
a0 = sA[r0][k]     │  sum00 += a0*b0  │  sum01 += a0*b1  │  <-- `a0` reused twice from Register!
                   ├──────────────────┼──────────────────┤
a1 = sA[r0+1][k]   │  sum10 += a1*b0  │  sum11 += a1*b1  │  <-- `a1` reused twice from Register!
                   └──────────────────┴──────────────────┘
                     (b0 reused 2x!)    (b1 reused 2x!)

• Shared Memory Reads per k step : Only 4 floats (a0, a1, b0, b1) instead of 8! (Cut in HALF!)
• Threads per 32x32 Tile Block   : 1,024 / 4 = 256 threads (16x16) -> 4 Blocks fit per SM!
```

---

### 4.3 Side-by-Side Detailed Dry Run & CUDA Mapping for `V2` (1 Single Thread `T(0,0)` Solving All 4 Cells `70, 30, 39, 15`!)

```text
DETAILED DRY RUN (1 Thread Solving Entire 2x2 Box!):      CUDA CODE + HARDWARE RESOURCE (Side-by-Side):
====================================================      =============================================
1. Thread (ty=0, tx=0) owns 2x2 corner (r0=0, c0=0):--->  int r0=ty*2;               // [Reg: Local Row 0]
   Rows: r0=0, r0+1=1  |  Cols: c0=0, c0+1=1              int c0=tx*2;               // [Reg: Local Col 0]
                                                          int row=blockIdx.y*TILE+r0;// [Reg: Global Row]
                                                          int col=blockIdx.x*TILE+c0;// [Reg: Global Col]

2. 4 Private Accumulator Registers in 1 Thread:     --->  float sum00=0.0f,sum01=0.0f; // [Regs: Top 2 cells]
   sum00=0, sum01=0, sum10=0, sum11=0                     float sum10=0.0f,sum11=0.0f; // [Regs: Bot 2 cells]

=======================================================================================================
ROUND 1: Tukda t = 0 (1 Thread loads 4 floats into sA & sB, then computes 4 cells in Registers)
=======================================================================================================
STEP 4.1 (Load 2x2 = 4 floats into sA & sB):        --->  int a_col=t*TILE+c0;
  sA = [ 1   2 ]        sB = [ 5   1 ]                    int b_row=t*TILE+r0;
       [ 2   1 ]             [ 6   2 ]                    sA[r0][c0]=A(row,a_col);     sA[r0][c0+1]=A(row,a_col+1);
                                                          sA[r0+1][c0]=A(row+1,a_col); sA[r0+1][c0+1]=A(row+1,a_col+1);
                                                          sB[r0][c0]=B(b_row,col);     sB[r0][c0+1]=B(b_row,col+1);
                                                          sB[r0+1][c0]=B(b_row+1,col); sB[r0+1][c0+1]=B(b_row+1,col+1);
                                                          __syncthreads();
-------------------------------------------------------------------------------------------------------
STEP 4.2 (Inner Loop k = 0, then k = 1 in Registers):-->  for(int k=0;k<TILE;k++){
                                                              float a0=sA[r0][k],   a1=sA[r0+1][k]; // [2 Regs]
  [Sub-step k = 0]:                                           float b0=sB[k][c0],   b1=sB[k][c0+1]; // [2 Regs]
  • Load 4 Regs from SRAM: a0=1, a1=2 | b0=5, b1=1
  • 4 Register Multiply-Adds:                                 sum00+=a0*b0;  // 1 * 5 = 5
    sum00 = 1*5 = 5   |  sum01 = 1*1 = 1                      sum01+=a0*b1;  // 1 * 1 = 1   (a0 Reused!)
    sum10 = 2*5 = 10  |  sum11 = 2*1 = 2                      sum10+=a1*b0;  // 2 * 5 = 10  (b0 Reused!)
                                                              sum11+=a1*b1;  // 2 * 1 = 2   (a1,b1 Reused!)
  [Sub-step k = 1]:                                       }
  • Load 4 Regs from SRAM: a0=2, a1=1 | b0=6, b1=2        __syncthreads();
  • 4 Register Multiply-Adds:
    sum00 = 5  + (2*6) = 17  |  sum01 = 1 + (2*2) = 5
    sum10 = 10 + (1*6) = 16  |  sum11 = 2 + (1*2) = 4

=======================================================================================================
ROUND 2: Tukda t = 1 (Overwrites sA & sB with t = 1 numbers, accumulates into sum00..sum11)
=======================================================================================================
STEP 4.3 (Load t = 1 numbers into sA & sB):
  sA = [ 3   4 ]        sB = [ 7   3 ]
       [ 1   2 ]             [ 8   4 ]
-------------------------------------------------------------------------------------------------------
STEP 4.4 (Inner Loop k = 0, then k = 1 in Registers):
  [Sub-step k = 0]:
  • Load 4 Regs from SRAM: a0=3, a1=1 | b0=7, b1=3
  • 4 Register Multiply-Adds:
    sum00 = 17 + (3*7) = 38  |  sum01 = 5  + (3*3) = 14
    sum10 = 16 + (1*7) = 23  |  sum11 = 4  + (1*3) = 7

  [Sub-step k = 1]:
  • Load 4 Regs from SRAM: a0=4, a1=2 | b0=8, b1=4
  • 4 Register Multiply-Adds:
    sum00 = 38 + (4*8) = 70  |  sum01 = 14 + (4*4) = 30
    sum10 = 23 + (2*8) = 39  |  sum11 = 7  + (2*4) = 15

=======================================================================================================
STEP 4.5 (Write All 4 Register Answers to VRAM C):   --->  C(row,  col  )=sum00; // Writes 70
                                                           C(row,  col+1)=sum01; // Writes 30
                                                           C(row+1,col  )=sum10; // Writes 39
                                                           C(row+1,col+1)=sum11; // Writes 15
```

---

### 4.4 Exact Hardware Resource Saturation in `V2` on Tesla T4 (`sm_75`)

```text
========================================================================================================
TESLA T4 (sm_75) HARDWARE RESOURCE UTILIZATION FOR V2 (`matmul_v2_reg_tiled`)
========================================================================================================
1. Thread & Block Concurrency:
   • Threads per Block        : 16 x 16 = 256 Threads (8 Warps)
   • Active Blocks per SM     : 4 Blocks / SM (Up from 1 Block/SM in V1 32x32!)
   • Active Threads per SM    : 4 Blocks * 256 Threads = 1,024 / 1,024 Threads (100.0% Occupancy!)

2. Shared Memory Utilization:
   • Shared Memory per Block  : sA[32][32] (4 KB) + sB[32][32] (4 KB) = 8,192 Bytes (8 KB)
   • Shared Memory per SM     : 4 Blocks * 8 KB = 32 KB / 64 KB (50.0% of SM Shared Memory Pool)

3. 100.0% Register File Saturation (Zero Waste, Zero Spills):
   • Registers per Thread     : 64 Registers (reported by `nvcc -Xptxas -v`, 0 bytes spill)
   • Total Registers per SM   : 1,024 Threads * 64 Registers = 65,536 / 65,536 Registers (100.0% Used!)
   • Hardware Boundary Note   : If the kernel used even 1 more register (65 regs/thread),
                                1,024 * 65 = 66,560 > 65,536, which would drop Blocks/SM from 4 to 3!
========================================================================================================
```

---
---

## 5. Master Benchmark & Hardware Resource Summary on NVIDIA Tesla T4 (`N = 1024 x 1024`)

* **Total FLOPs per Run:** $2 \times N^3 = 2 \times 1024^3 = 2,147,483,648$ FLOPs (`2.147 GFLOPs`).
* **Throughput Formula:** $\text{GFLOPS} = \frac{2 \times N^3}{\text{avg\_ms} \times 10^6}$.

| Kernel Version | Cells / Thread | Threads / Block | Blocks / SM | Active Threads / SM | Shared Mem / Block (`SM`) | Registers / Thread (`SM Total`) | VRAM Read Traffic | Avg Time (ms) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`V0`: Naive Global (`16x16`)** | `1` | `256` | `4` | `1024 / 1024` (`100%`) | `0 B` (`0 KB`) | `52` (`53,248 / 65,536`) | `8.38 GB` | `4.02 ms` | `534.2 GFLOPS` |
| **`V1`: Shared Tiled (`16x16`, `CHECK_BOUNDS=true`)** | `1` | `256` | `4` | `1024 / 1024` (`100%`) | `2,048 B` (`8 KB`) | `37` (`37,888 / 65,536`) | `524 MB` | `5.44 ms` | `394.7 GFLOPS` |
| **`V1`: Shared Tiled (`16x16`, `CHECK_BOUNDS=false`)** | `1` | `256` | `4` | `1024 / 1024` (`100%`) | `2,048 B` (`8 KB`) | `38` (`38,912 / 65,536`) | `524 MB` | `2.32 ms` | `923.8 GFLOPS` |
| **`V1`: Shared Tiled (`32x32`, `CHECK_BOUNDS=false`)** | `1` | `1024` | `1` | `1024 / 1024` (`100%`) | `8,192 B` (`8 KB`) | `41` (`41,984 / 65,536`) | `262 MB` | `2.03 ms` | `1,059.5 GFLOPS` |
| **`V2`: `32x32` Shared + `2x2` Register Tile** | **`4` (`2x2`)** | **`256`** | **`4`** | **`1024 / 1024` (`100%`)** | **`8,192 B` (`32 KB`)** | **`64` (`65,536 / 65,536`)** | **`262 MB` (+2x SRAM reuse)** | **`1.14 ms`** | **`1,875.1 GFLOPS` (`1.88 TFLOPS`)** |
