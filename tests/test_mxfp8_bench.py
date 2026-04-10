import sys
import torch

sys.path.insert(0, '/mair/team-sys/jangwoong/DeepGEMM/tests')

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import mxfp8_quantize_output

from generators import (
    KernelType, QuantConfig, MajorTypeAB, get_ue8m0_usage, generate_normal,
)


def dequantize_mxfp8(fp8_data, e8m0_scales_uint8):
    m, n = fp8_data.shape
    scale = torch.pow(2.0, e8m0_scales_uint8.float() - 127.0)
    return (fp8_data.float().view(m, n // 32, 32) * scale.unsqueeze(2)).view(m, n)


def test_mxfp8_bench():
    print('='*80)
    print('MXFP8 Benchmark: Path1 (fp8_gemm_nt_mxfp8out) vs Path2 (fp8_gemm_nt + quantize)')
    print('='*80)

    arch = get_arch_major()
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant_config = QuantConfig()
    disable_cast = not use_ue8m0

    test_shapes = [
        (1, 7168, 2048),
        (128, 7168, 2048),
        (256, 7168, 2048),
        (4096, 7168, 2048),
        (128, 4096, 7168),
        (256, 4096, 7168),
    ]

    print(f'\nArch: SM{arch}0, use_ue8m0={use_ue8m0}')
    print(f'{"M":>6} {"N":>6} {"K":>6} | {"Path1(mxfp8out)":>15} {"Path2(bf16+quant)":>17} {"Speedup":>8} | {"Diff":>10}')
    print('-'*95)

    num_iters = 50

    for m, n, k in test_shapes:
        if n % 32 != 0:
            continue

        # Generate inputs
        a, b, c, d_dummy, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, KernelType.Kernel1D1D,
            use_ue8m0=use_ue8m0, quant_config=quant_config)

        # --- Path 1: fp8_gemm_nt_mxfp8out (my API) ---
        d_fp8_1 = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
        d_sf_1 = torch.empty((m, n // 32), device='cuda', dtype=torch.uint8)

        def path1():
            deep_gemm.fp8_gemm_nt_mxfp8out(a, b, d_fp8_1, d_sf_1, disable_ue8m0_cast=disable_cast)

        # --- Path 2: fp8_gemm_nt (BF16 out) + mxfp8_quantize_output (manual) ---
        d_bf16_2 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

        def path2():
            deep_gemm.fp8_gemm_nt(a, b, d_bf16_2, disable_ue8m0_cast=disable_cast)
            return mxfp8_quantize_output(d_bf16_2, block_size=32)

        # Warmup
        path1()
        fp8_2, sf_2 = path2()

        # Accuracy: dequantize both and compare
        deq_1 = dequantize_mxfp8(d_fp8_1, d_sf_1)
        deq_2 = dequantize_mxfp8(fp8_2, sf_2)
        diff = calc_diff(deq_1, deq_2)

        # --- Performance ---
        torch.cuda.synchronize()
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s1.record()
        for _ in range(num_iters):
            path1()
        e1.record()
        torch.cuda.synchronize()
        t1 = s1.elapsed_time(e1) / num_iters

        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s2.record()
        for _ in range(num_iters):
            path2()
        e2.record()
        torch.cuda.synchronize()
        t2 = s2.elapsed_time(e2) / num_iters

        speedup = t2 / t1 if t1 > 0 else 0
        print(f'{m:6} {n:6} {k:6} | {t1:13.3f}ms {t2:15.3f}ms {speedup:7.2f}x | {diff:10.6f}')

    # --- torch.compile ---
    print(f'\n{"="*80}')
    print('torch.compile test (m=256, n=7168, k=2048):')
    print(f'{"="*80}')

    m, n, k = 256, 7168, 2048
    a, b, c, d_dummy, ref_d = generate_normal(
        m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        False, torch.bfloat16, KernelType.Kernel1D1D,
        use_ue8m0=use_ue8m0, quant_config=quant_config)

    d_fp8_c = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
    d_sf_c = torch.empty((m, n // 32), device='cuda', dtype=torch.uint8)

    def fn_mxfp8out():
        deep_gemm.fp8_gemm_nt_mxfp8out(a, b, d_fp8_c, d_sf_c, disable_ue8m0_cast=disable_cast)

    d_bf16_c = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    def fn_manual():
        deep_gemm.fp8_gemm_nt(a, b, d_bf16_c, disable_ue8m0_cast=disable_cast)
        return mxfp8_quantize_output(d_bf16_c, block_size=32)

    for label, fn in [('mxfp8out', fn_mxfp8out), ('manual', fn_manual)]:
        for fullgraph in (True, False):
            try:
                compiled = torch.compile(fn, fullgraph=fullgraph)
                for _ in range(3):
                    compiled()
                torch.cuda.synchronize()

                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(num_iters):
                    compiled()
                e.record()
                torch.cuda.synchronize()
                t_compiled = s.elapsed_time(e) / num_iters

                # Baseline
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                s.record()
                for _ in range(num_iters):
                    fn()
                e.record()
                torch.cuda.synchronize()
                t_base = s.elapsed_time(e) / num_iters

                sp = t_base / t_compiled if t_compiled > 0 else 0
                print(f'  {label} fullgraph={fullgraph}: compiled={t_compiled:.3f}ms base={t_base:.3f}ms speedup={sp:.2f}x')
                break
            except Exception as ex:
                print(f'  {label} fullgraph={fullgraph}: FAILED — {str(ex)[:120]}')
                if fullgraph:
                    continue
                else:
                    break

    print('\nDone!')


if __name__ == '__main__':
    test_mxfp8_bench()
