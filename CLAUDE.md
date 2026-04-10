# MXFP8 Epilogue Fusion — SM100 (B200)

## Goal
SM100 커널 에필로그에서 FP32 accumulator를 **커널 내부에서** FP8 + E8M0 scale로 직접 변환하여 HBM에 쓴다.
BF16 중간 출력 + 별도 quantize 커널 없이, 1-kernel로 끝낸다.

## 현재 상태: 성공 (2026-04-10)

B200 (SM100)에서 epilogue fusion + TMA store. Correctness + Performance 모두 통과.

baseline: `torch.compile(fp8_gemm_nt + mxfp8_quantize_output, fullgraph=True)` — GEMM과 quantize를 하나의 compiled fx graph로.

### Normal GEMM (fp8_gemm_nt_mxfp8out)
```
     M      N      K |   Fused | BL compiled | Speedup |     Diff
     1   7168   2048 | 0.149ms |    0.199ms  |  1.34x  | 0.00011
   128   7168   2048 | 0.159ms |    0.214ms  |  1.34x  | 0.00013
   256   7168   2048 | 0.155ms |    0.232ms  |  1.50x  | 0.00013
  4096   7168   2048 | 0.191ms |    0.271ms  |  1.42x  | 0.00013
   128   4096   7168 | 0.159ms |    0.247ms  |  1.55x  | 0.00013
   256   4096   7168 | 0.168ms |    0.273ms  |  1.62x  | 0.00013
```

### Grouped GEMM (m_grouped_fp8_gemm_nt_contiguous_mxfp8out)
```
  G      M      N      K |   Fused | BL compiled | Speedup |     Diff
  4  30464   7168   2048 | 0.493ms |    0.661ms  |  1.34x  | 0.00013
  8  36864   7168   2048 | 0.576ms |    0.804ms  |  1.40x  | 0.00013
  4  34688   4096   7168 | 0.935ms |    1.620ms  |  1.73x  | 0.02003
  8  32256   4096   7168 | 0.888ms |    1.539ms  |  1.73x  | 0.02003
 48  63232   1280   4096 | 0.401ms |    0.851ms  |  2.12x  | 0.02578
```

모든 shape에서 `fullgraph=True`. Diff ~0.02는 fused가 FP32 TMEM에서 직접 quantize하고 baseline은 BF16 truncation 후 quantize하기 때문. **fused가 더 정확함**.

### Kernel-level profiler (G=48 shape)
```
Fused kernel:    ~273us
Baseline kernel: ~273us (동등)
```
Fused가 e2e에서 빠른 이유: baseline은 GEMM kernel + compiled Triton quantize kernel = 2 kernels. Fused는 1 kernel.

## 핵심 해결책
기존 epilogue의 s-loop/TMA pipeline을 **완전 우회**. `if constexpr (kIsMXFP8Output)`로 분기:
- 기존 path: TMEM → SMEM (swizzled) → TMA store → HBM
- MXFP8 path: TMEM → registers (32개 연속 N값) → quantize → SMEM (FP8) → TMA store → HBM

MXFP8 전용 s-loop: 기존 pipeline 구조 (wait → write → fence → sync → TMA → arrive) 동일, 내부 로직만 다름. Scale은 gmem direct write (너무 작아서 TMA 비효율).

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
