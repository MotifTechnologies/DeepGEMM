# MXFP8 Epilogue Fusion: FP8 Output from GEMM Kernel

## Motivation

Current flow for consecutive MXFP8 GEMMs (e.g. MoE expert layers):

```
GEMM epilogue: FP32 accum → SMEM → TMA store → HBM (BF16)    2 bytes/elem write
quantize kern: HBM (BF16) → FP8 + scale → HBM                2 bytes read + ~1 byte write
next GEMM:     HBM (FP8) → ...                                1 byte read
```

Total HBM traffic per element between two GEMMs: **2 (write) + 2 (read) + 1 (write) + 1 (read) = 6 bytes**.

With epilogue fusion:

```
GEMM epilogue: FP32 accum → MXFP8 quantize in-kernel → HBM (FP8 + scale)  ~1 byte write
next GEMM:     HBM (FP8) → ...                                              1 byte read
```

Total: **~2 bytes**. **3x bandwidth reduction** between consecutive GEMMs.

This matters because MoE expert GEMMs often run with small M (few tokens per expert), making them memory-bandwidth-bound.

## Current Epilogue Code

### SM90 (`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`)

```
Line 44:  DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>)  — accum always FP32
Line 59:  SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float)        — SMEM for output tile
```

Epilogue flow (lines 350-367):

```cuda
// 1. FP32 registers → SMEM
st_shared(smem_d_0 + i * 4, {final_accum[i*4+0], final_accum[i*4+1]});
st_shared(smem_d_1 + i * 4, {final_accum[i*4+2], final_accum[i*4+3]});

// 2. TMA store: SMEM (FP32) → HBM (BF16 via TMA descriptor)
cute::SM90_TMA_REDUCE_ADD_2D::copy(&tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N, ...);
```

### SM100 (`deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh`)

Epilogue flow (lines 501-544):

```cuda
// 1. TMEM → registers → SMEM (BF16 path, lines 508-519)
cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, values[0..7]);
st_shared(smem_ptr,
    cast_into_bf16_and_pack(values[0], values[1]),  // FP32 → BF16x2
    cast_into_bf16_and_pack(values[2], values[3]),
    cast_into_bf16_and_pack(values[4], values[5]),
    cast_into_bf16_and_pack(values[6], values[7]));

// 2. TMA store: SMEM → HBM
cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
```

`cast_into_bf16_and_pack` is defined in `common/utils.cuh:156`:

```cuda
auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
return *reinterpret_cast<int*>(&bf16x2);
```

### Host Side (`csrc/apis/gemm.hpp`)

```
Line 169:  DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
```

TMA descriptor creation (`csrc/jit_kernels/impls/runtime_utils.hpp:199`):

```cpp
static CUtensorMap make_tma_cd_desc(const torch::Tensor& t,
    const int& shape_m, const int& shape_n,
    const int& block_m, const int& block_n,
    const int& outer_stride, const int& num_groups, ...);
```

## Proposed Change

Replace `cast_into_bf16_and_pack` with MXFP8 quantization in the epilogue. Output two tensors: FP8 data + E8M0 scale factors.

### MXFP8 Quantization Math

For each block of 32 contiguous elements along N:

```
max_abs = max(|x[0]|, |x[1]|, ..., |x[31]|)
E = floor(log2(max_abs / 448.0)) + 127        // E8M0 exponent, clamped to [0, 254]
scale = 2^(E - 127)
fp8[i] = saturate_e4m3(x[i] / scale)          // clamp to [-448, 448]
```

Output: `fp8_data (M, N)` as `float8_e4m3fn` + `scale (M, N/32)` as `uint8` (E8M0).

## Files to Modify

### 1. Kernel Epilogue (`.cuh` files)

**SM100** — `sm100_fp8_gemm_1d1d.cuh` lines 508-519:

Replace:
```cuda
st_shared(smem_ptr,
    cast_into_bf16_and_pack(values[0], values[1]), ...);
```

With (pseudocode):
```cuda
// Convert FP32 → FP8 with block-wise scale
float abs_vals[8];
for (int i = 0; i < 8; i++)
    abs_vals[i] = fabsf(*reinterpret_cast<float*>(&values[i]));

// Warp reduction to get max over 32 elements (see "Hard Part" below)
float block_max = warp_reduce_max_32(abs_vals, lane_idx);

// Compute E8M0 scale
uint8_t E = compute_e8m0_exponent(block_max);
float scale = exp2f((float)E - 127.0f);
float inv_scale = 1.0f / scale;

// Cast to FP8
uint8_t fp8_packed[8];
for (int i = 0; i < 8; i++)
    fp8_packed[i] = float_to_e4m3(values[i] * inv_scale);

// Store FP8 data to SMEM (4 bytes = 4 fp8 elements at a time)
st_shared(smem_fp8_ptr, pack4_fp8(fp8_packed));

// Store scale to separate SMEM region (one uint8 per 32-element block)
if (is_block_leader)
    st_shared(smem_scale_ptr, E);
```

**SM90** — `sm90_fp8_gemm_1d1d.cuh` lines 354-356:

Same logic applied to the `st_shared(smem_d_0, final_accum)` path.

### 2. SMEM Layout

Current:
```
SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(bfloat16)   // 2 bytes/elem
```

Changed to:
```
SMEM_D_FP8_SIZE   = BLOCK_M * BLOCK_N * sizeof(fp8)              // 1 byte/elem
SMEM_D_SCALE_SIZE = BLOCK_M * (BLOCK_N / 32) * sizeof(uint8)     // 1/32 byte/elem
```

Net SMEM usage drops ~2x, which may allow larger tiles or more pipeline stages.

### 3. TMA Descriptors (`.hpp` host code)

Need two TMA descriptors instead of one:

```cpp
// csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp (and sm100 variant)

// Current:
const auto& tensor_map_cd = make_tma_cd_desc(d, m, n, block_m, block_n, ...);

// Changed:
const auto& tensor_map_d_fp8   = make_tma_cd_desc(d_fp8, m, n, block_m, block_n, ...);
const auto& tensor_map_d_scale = make_tma_sf_desc(d_scale, m, n/32, block_m, block_n/32, ...);
```

Kernel launch now passes both descriptors.

### 4. Host API (`csrc/apis/gemm.hpp`)

```cpp
// Line 169 — change from:
DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

// To: accept (fp8_tensor, scale_tensor) pair as output
DG_HOST_ASSERT(d.scalar_type() == torch::kFloat8_e4m3fn);
DG_HOST_ASSERT(d_scale.scalar_type() == torch::kUInt8);
```

New Python-level API:

```python
# Current:
out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous((qA, qA_sf), (qB, qB_sf), out, ...)

# New:
out_fp8   = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=device)
out_scale = torch.empty(M, N // 32, dtype=torch.uint8, device=device)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous_fp8out(
    (qA, qA_sf), (qB, qB_sf), (out_fp8, out_scale), ...)
```

### 5. Caller Side (`llm_training/motif/quantization/mxfp8_ops.py`)

```python
# Current (line 161-168):
qx, qx_sf = mxfp8_quantize_rowwise(x)           # separate quantize kernel
out = torch.empty(M, N, dtype=torch.bfloat16)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(...)   # BF16 output

# New:
qx, qx_sf = prev_gemm_fp8_output                  # already FP8 from previous GEMM
out_fp8, out_sf = deep_gemm.m_grouped_fp8_gemm_nt_contiguous_fp8out(...)
# Pass (out_fp8, out_sf) directly to next GEMM — no separate quantize needed
```

## Hard Part: Warp-Level 32-Element Max Reduction

The MXFP8 scale requires `max(|x[0..31]|)` over 32 contiguous N-dimension elements. But in the epilogue, these 32 elements are distributed across threads according to the WGMMA register layout.

### SM90 Register Layout

Each thread in a warpgroup holds accumulator fragments. For BLOCK_N=128:
- `WGMMA::kNumAccum` values per thread
- Thread `t` holds elements at specific N positions determined by the WGMMA tile layout
- Adjacent N elements may be in different threads within the same warp

Solution: `__shfl_xor_sync` to exchange values within a warp:

```cuda
// Pseudocode: reduce max over 32 N-contiguous elements
float local_max = fabsf(my_value);
for (int delta = 1; delta < 32; delta *= 2)
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, delta));
// Now all 32 threads in the warp have the same max
```

The exact shuffle pattern depends on the WGMMA accumulator layout mapping (which N-indices map to which lanes). This needs to be derived from the `WGMMA::M` and accumulator indexing in the kernel.

### SM100 TMEM Layout

On SM100, accumulators live in TMEM. The epilogue already loads from TMEM into registers via `SM100_TMEM_LOAD_32dp32b8x`. The 8 values loaded per thread are at specific TMEM addresses that follow a swizzled layout. The 32-element reduction needs to account for this swizzle pattern.

## Risks and Open Questions

1. **Register pressure**: FP8 quantization adds temporary registers for abs values, max reduction, and scale computation. Current kernels run at `__launch_bounds__(N, 1)` — verify occupancy is not hurt.

2. **Numerical equivalence**: Block-32 MXFP8 scale from GEMM epilogue FP32 vs. from a separate quantize kernel operating on BF16 will produce slightly different results (the separate kernel quantizes BF16, the fused path quantizes FP32 — fused is actually more accurate).

3. **Scale format compatibility**: DeepGEMM's input scale path expects float32 scales (which it converts internally to E8M0). If the output produces native E8M0 uint8, the next GEMM can skip the float32→E8M0 conversion — additional speedup.

4. **Non-GEMM consumers**: Some outputs go to non-GEMM ops (activation functions, residual connections) that need BF16/FP32. Need a flag or separate API to choose BF16 vs FP8 output mode.

5. **Backward pass compatibility**: The backward dX GEMM (`nn_contiguous`) also outputs BF16 that gets quantized for the next backward GEMM. Same optimization applies there.

## Implementation Order

1. **Prototype on SM100 first** — the `cast_into_bf16_and_pack` branch (line 508-519) is a clean insertion point
2. Add `cast_into_fp8_and_scale` utility function in `common/utils.cuh` alongside existing `cast_into_bf16_and_pack`
3. Add new `cd_dtype_t = float8_e4m3` branch in the epilogue `if constexpr` chain
4. Update host-side TMA descriptor creation and API
5. Port to SM90 (different epilogue structure, same quantization logic)
6. Update Python wrappers in `mxfp8_ops.py`

---

## Current Implementation (SM90)

SM90 prototype implemented and compiles. API: `deep_gemm.fp8_gemm_nt_mxfp8out((a, a_sf), (b, b_sf), d_fp8, d_sf)`.

K-loop 완료 후 FP32 accumulator 레지스터에서 직접 quantize → FP8 SMEM → TMA store → HBM. `ceil` 기반 E8M0 exponent, warp shuffle 2회로 32-element max reduction.

Constraints: `N % 32 == 0`, no accumulation, SM90 only, NT layout only. GPU 미검증.

### Prerequisites for register-level quantize (SM90)

1. WGMMA 64xNx32 layout 사용 (Hopper)
2. N이 32의 배수
3. 32-element group이 warp 내 4 thread에 완전히 분배됨

3번은 WGMMA Z-pattern (`col_idx = lane_idx % 4`, 각 thread가 8개씩)으로 확인됨. Colfax tutorial + 기존 `st_shared` 코드 역산으로 검증.

### Known weaknesses

1. **Byte-granularity SMEM writes** — 1바이트씩 write. `pack_fp8x4`로 4바이트 단위 store 필요.
2. **No TMA swizzle** — FP8 output에 swizzle=0. Bank conflict 발생.
3. **Heuristic 미최적화** — BF16 proxy로 config 선택. FP8 output의 4x 작은 SMEM 미활용.
4. **TMA store 2회** — scale tensor가 너무 작아 TMA 효율 낮음.
5. **GPU 미검증**.

---

## SM100 (Blackwell / GB200) 구현

### TMEM 매핑 분석 결과

`tmem_addr` 계산식에서 N-position을 역산:

```cuda
tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N
          + w * BLOCK_N
          + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
```

FP32 path에서 `kNumElemsPerBankGroup=4`, `STORE_BLOCK_N = kSwizzleCDMode / 4`. `kSwizzleCDMode=128`이면 `STORE_BLOCK_N=32`.

i-loop가 `STORE_BLOCK_N / 4 = 8`번 반복, 매번 4개 연속 N값 로드. 한 thread가 한 s-iteration에서 보는 N-positions:

```
i=0: s*32 + 0,1,2,3
i=1: s*32 + 4,5,6,7
...
i=7: s*32 + 28,29,30,31
```

**한 thread가 32개 연속 N값을 전부 소유. lane_idx는 M-row만 결정.**

### SM90 vs SM100 비교

| | SM90 (H100) | SM100 (GB200) |
|---|---|---|
| accumulator 위치 | 레지스터 | TMEM |
| 32-element 분배 | 4 thread × 8개 | **1 thread × 32개** |
| shuffle 필요 | 2회 | **0회** |
| reduction | warp shuffle | **thread-local** |

SM100이 SM90보다 **더 쉬움**. TMEM column index가 N-position에 직접 대응하고, 한 thread가 한 MXFP8 group을 혼자 처리.

### 에필로그 흐름

```
기존 BF16:  i-loop { TMEM load 4 vals → bf16 pack → SMEM } → TMA store
MXFP8:     i-loop { TMEM load 4 vals → register 누적 (32개) + max 추적 }
           → E8M0 exponent 계산 + inv_scale
           → FP8 변환 → SMEM(FP8) write + scale write
           → TMA store ×2 (FP8 data + E8M0 scales)
```

Pseudocode:

```cuda
float group_vals[32];
float group_max = 0;

for (uint32_t i = 0; i < STORE_BLOCK_N / 4; ++i) {
    SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, v[0], v[1], v[2], v[3]);
    fence_view_async_tmem_load();
    for (int j = 0; j < 4; j++) {
        float val = *reinterpret_cast<float*>(&v[j]);
        group_vals[i*4 + j] = val;
        group_max = fmaxf(group_max, fabsf(val));
    }
}

uint8_t e8m0 = compute_e8m0_exponent(group_max);
float inv_scale = exp2f(127.0f - e8m0);

for (int k = 0; k < 32; k++)
    smem_fp8[lane_idx * BLOCK_N + s * 32 + k] = float_to_fp8_e4m3_sat(group_vals[k] * inv_scale);
smem_sf[lane_idx * (BLOCK_N / 32) + s] = e8m0;
```

### SM100 epilogue fusion 시도 및 현재 상태

커널 내부에서 TMEM → register → FP8 quantize → global memory direct write를 시도했으나, SM100의 epilogue pipeline (TMA store staging, SMEM swizzle, barrier 동기화)이 BF16/FP32 config와 강하게 결합되어 있어 `cd_dtype_t` override만으로는 동작하지 않음.

구체적 문제들:
1. **SMEM size 불일치** — BF16 proxy config의 `swizzle_cd_mode`를 override하면 SMEM_CD_SIZE가 바뀌어 총 SMEM이 하드웨어 한계 초과
2. **TMA store pipeline 비동기** — MXFP8는 32개 N-element 모일 때만 store하지만, pipeline은 매 s-iteration마다 store를 기대
3. **STORE_BLOCK_N 불일치** — BF16 config의 block_n이 16이면 STORE_BLOCK_N < 32, MXFP8 group과 불일치

**현재 해결**: 2-kernel 방식으로 correctness 검증 완료.

```
Step 1: 기존 fp8_gemm_nt → BF16 output (기존 SM100 커널 그대로 사용)
Step 2: Python mxfp8_quantize_output → FP8 data + E8M0 scales
```

**B200 (SM100) 검증 결과** (2026-04-09):

```
m=128, n=256, k=128: diff=0.00129  ✓
m=128, n=128, k=128: diff=0.00126  ✓
m=256, n=512, k=128: diff=0.00129  ✓
All tests passed (diff < 0.01)
```

### 향후 epilogue fusion을 위한 방향

SM100 커널 epilogue를 수정하려면:
1. `cd_dtype_t` proxy가 아닌, **별도 template bool `kIsMXFP8Output`를 독립 파라미터로** 추가하여 기존 BF16/FP32 상수들과 분리
2. MXFP8 전용 epilogue loop 구조 설계 (기존 s-loop/TMA pipeline과 별개)
3. 또는 SMEM 경유 없이 **TMEM → register → global memory direct write** 전용 path (TMA pipeline 완전 우회)

### Not yet implemented

- SM100 epilogue fusion (1-kernel)
- Grouped GEMM variants (`m_grouped_*`, `k_grouped_*`)
- NN/TN/TT layout aliases
- Accumulation (`c != None`)
- Backward pass integration
