# MXFP8 Epilogue Fusion — SM100 (B200)

## Goal
SM100 커널 에필로그에서 FP32 accumulator를 **커널 내부에서** FP8 + E8M0 scale로 직접 변환하여 HBM에 쓴다.
BF16 중간 출력 + 별도 quantize 커널 없이, 1-kernel로 끝낸다.

## 현재 상태: 성공 (2026-04-10)

B200 (SM100)에서 epilogue fusion 동작 확인. Correctness + Performance 모두 통과.

### Normal GEMM (fp8_gemm_nt_mxfp8out)
baseline: fp8_gemm_nt BF16 out + torch.compile(mxfp8_quantize_output, fullgraph=True)
```
     M      N      K |   Fused | BL gemm BL quant BL total | Speedup |     Diff
     1   7168   2048 | 0.135ms | 0.159ms 0.066ms  0.225ms  |  1.67x  | 0.00013
   128   7168   2048 | 0.168ms | 0.137ms 0.073ms  0.210ms  |  1.25x  | 0.00013
   256   7168   2048 | 0.160ms | 0.154ms 0.076ms  0.230ms  |  1.43x  | 0.00013
  4096   7168   2048 | 0.230ms | 0.183ms 0.184ms  0.367ms  |  1.60x  | 0.00013
   128   4096   7168 | 0.154ms | 0.157ms 0.075ms  0.232ms  |  1.50x  | 0.00013
   256   4096   7168 | 0.154ms | 0.139ms 0.073ms  0.212ms  |  1.38x  | 0.00013
```

### Grouped GEMM (m_grouped_fp8_gemm_nt_contiguous_mxfp8out)
```
  G      M      N      K |   Fused | BL gemm BL quant BL total | Speedup |     Diff
  4  31616   7168   2048 | 0.773ms | 0.481ms 1.080ms  1.561ms  |  2.02x  | 0.00013
  8  32256   7168   2048 | 0.779ms | 0.488ms 1.085ms  1.573ms  |  2.02x  | 0.00013
  4  33920   4096   7168 | 1.006ms | 0.924ms 0.780ms  1.704ms  |  1.69x  | 0.02007
  8  36864   4096   7168 | 1.151ms | 1.004ms 0.824ms  1.828ms  |  1.59x  | 0.02004
 48  64384   1280   4096 | 0.499ms | 0.412ms 0.405ms  0.817ms  |  1.64x  | 0.02574
```

Diff가 K=7168, K=4096에서 ~0.02인 이유: fused는 FP32 TMEM에서 직접 quantize, baseline은 BF16 truncation 후 quantize. BF16 중간 손실로 E8M0 exponent 경계가 달라짐. **fused가 더 정확함**.

Grouped GEMM도 동일 커널의 MXFP8 epilogue가 그대로 적용됨. 추가 코드 수정 없이 동작.

## 핵심 해결책
기존 epilogue의 s-loop/TMA pipeline을 **완전 우회**. `if constexpr (kIsMXFP8Output)`로 분기:
- 기존 path: TMEM → SMEM (swizzled) → TMA store → HBM
- MXFP8 path: TMEM → registers (32개 연속 N값) → quantize → global memory direct write

TMA pipeline (advance_store_pipeline, tma_store_wait, tma_store_fence, smem swizzle)을 일절 사용하지 않음.

## 이전 실패 원인 정리 (7번 실패)
1. cd_dtype override → STORE_BLOCK_N / kNumElemsPerBankGroup / SMEM_CD_SIZE 연쇄 불일치
2. swizzle_cd_mode 강제 → SMEM 크기 초과 (232KB 한계)
3. SMEM size delta 재계산 → cuFuncSetAttribute reject
4. swizzle override 제거 + 여러 s-iteration 누적 → SMEM overflow
5. 매 s-iteration TMA store → TMA pipeline 비동기 (advance_store_pipeline 꼬임)
6. TMA 제거 + gmem direct write → 빈 TMA descriptor prefetch crash
7. valid TMA placeholder + tma_store_wait/fence skip → 여전히 illegal access (barrier sync 꼬임)

**근본 원인**: 기존 epilogue에 분기를 끼워넣으면 s-loop/TMA pipeline의 모든 요소가 연쇄적으로 깨짐.
**해결**: 기존 epilogue를 건드리지 않고, `if constexpr`로 **완전히 별도 코드 path**를 실행.

## 다음 최적화: TMA store 복구

### 문제
현재 MXFP8 epilogue는 global memory direct write (thread별 4-byte store). Profiler 결과:
- Fused kernel: 361.9us (GEMM + gmem direct write epilogue)
- Baseline GEMM kernel: 282.2us (GEMM + TMA store epilogue)
- 차이 80us = gmem direct write vs TMA store 비효율

### 목표
MXFP8 epilogue에서 TMA store를 사용하여 fused kernel 시간을 baseline GEMM과 비슷하게 (~280us) 만듦.

### "성공" 정의
- **baseline**: 기존 DeepGEMM `fp8_gemm_nt` (BF16 out) + `torch.compile(mxfp8_quantize_output, fullgraph=True)`
- **target**: `fp8_gemm_nt_mxfp8out` (TMA store 복구)
- **성공 조건**:
  1. target이 baseline보다 **성능이 빠름** (kernel-level에서도 비슷하거나 빠름)
  2. accuracy test 통과 (diff < 0.03)
- **비교 방법**: 둘 다 `torch.compile(fullgraph=True)` 적용. profiler trace로 kernel-level 비교.

### 결과: TMA store 복구 성공 (2026-04-10)

MXFP8 전용 s-loop: TMEM load → register quantize → FP8 SMEM write → TMA store.
기존 s-loop과 동일한 pipeline 구조 (wait → write → fence → sync → TMA → arrive), 코드는 별도.

Profiler kernel-level:
- 이전 (gmem direct write): Fused 361.9us vs Baseline 282.2us → fused가 80us 느림
- **현재 (TMA store): Fused ~273us ≈ Baseline ~273us → 동등**

e2e (torch.compile baseline 대비):
```
Normal:  1.43~2.04x speedup
Grouped: 2.27~3.59x speedup (G=48 M=64K: 0.408ms vs 0.925ms = 2.27x)
```
