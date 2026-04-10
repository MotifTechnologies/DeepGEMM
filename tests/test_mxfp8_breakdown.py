import sys
import torch

sys.path.insert(0, '/mair/team-sys/jangwoong/DeepGEMM/tests')

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import mxfp8_quantize_output

from generators import (
    KernelType, QuantConfig, MajorTypeAB, get_ue8m0_usage,
    generate_normal, generate_m_grouped_contiguous,
)


def bench(fn, num_iters=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / num_iters


def dequantize_mxfp8(fp8_data, e8m0_scales_uint8):
    m, n = fp8_data.shape
    scale = torch.pow(2.0, e8m0_scales_uint8.float() - 127.0)
    return (fp8_data.float().view(m, n // 32, 32) * scale.unsqueeze(2)).view(m, n)


def run_normal_gemm_test():
    print('='*100)
    print('[Normal GEMM] Accuracy + Performance Breakdown (with torch.compile)')
    print('='*100)

    arch = get_arch_major()
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant_config = QuantConfig()
    disable_cast = not use_ue8m0

    shapes = [
        (1, 7168, 2048),
        (128, 7168, 2048),
        (256, 7168, 2048),
        (4096, 7168, 2048),
        (128, 4096, 7168),
        (256, 4096, 7168),
    ]

    try:
        compiled_quant = torch.compile(mxfp8_quantize_output, fullgraph=True)
        compile_mode = 'fullgraph=True'
    except Exception:
        compiled_quant = torch.compile(mxfp8_quantize_output, fullgraph=False)
        compile_mode = 'fullgraph=False'

    _dummy = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)
    for _ in range(3):
        compiled_quant(_dummy, block_size=32)
    torch.cuda.synchronize()

    print(f'\nArch: SM{arch}0, torch.compile quantize: {compile_mode}')
    print(f'{"M":>6} {"N":>6} {"K":>6} | {"Fused":>7} | {"BL gem":>7} {"BL qnt":>7} {"BL tot":>7} | {"Spdup":>5} | {"Diff":>8}')
    print('-'*90)

    for m, n, k in shapes:
        a, b, c, d_dummy, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, KernelType.Kernel1D1D,
            use_ue8m0=use_ue8m0, quant_config=quant_config)

        # Fused
        d_fp8 = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
        d_sf = torch.empty((m, n // 32), device='cuda', dtype=torch.float8_e4m3fn)
        t_fused = bench(lambda: deep_gemm.fp8_gemm_nt_mxfp8out(
            a, b, d_fp8, d_sf, disable_ue8m0_cast=disable_cast))

        # Baseline gemm
        d_bf16 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        t_bl_gemm = bench(lambda: deep_gemm.fp8_gemm_nt(
            a, b, d_bf16, disable_ue8m0_cast=disable_cast))

        # Baseline quantize (compiled)
        deep_gemm.fp8_gemm_nt(a, b, d_bf16, disable_ue8m0_cast=disable_cast)
        torch.cuda.synchronize()
        for _ in range(3):
            compiled_quant(d_bf16, block_size=32)
        torch.cuda.synchronize()
        t_bl_quant = bench(lambda: compiled_quant(d_bf16, block_size=32))

        # Accuracy: fused vs baseline
        deep_gemm.fp8_gemm_nt_mxfp8out(a, b, d_fp8, d_sf, disable_ue8m0_cast=disable_cast)
        deep_gemm.fp8_gemm_nt(a, b, d_bf16, disable_ue8m0_cast=disable_cast)
        bl_fp8, bl_sf = mxfp8_quantize_output(d_bf16, block_size=32)
        deq_fused = dequantize_mxfp8(d_fp8, d_sf.view(torch.uint8))
        deq_bl = dequantize_mxfp8(bl_fp8, bl_sf)
        diff = calc_diff(deq_fused, deq_bl)

        t_bl_total = t_bl_gemm + t_bl_quant
        speedup = t_bl_total / t_fused if t_fused > 0 else 0

        print(f'{m:6} {n:6} {k:6} | {t_fused:5.3f}ms | {t_bl_gemm:5.3f}ms {t_bl_quant:5.3f}ms {t_bl_total:5.3f}ms | {speedup:4.2f}x | {diff:8.5f}')

    print()


def run_grouped_gemm_test():
    print('='*100)
    print('[Grouped GEMM] Accuracy + Performance Breakdown (with torch.compile)')
    print('='*100)

    arch = get_arch_major()
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant_config = QuantConfig()
    disable_cast = not use_ue8m0

    grouped_shapes = [
        # (num_groups, expected_m_per_group, n, k)
        (4, 8192, 7168, 2048),
        (8, 4096, 7168, 2048),
        (4, 8192, 4096, 7168),
        (8, 4096, 4096, 7168),
        (48, 1280, 1280, 4096),  # ~60K total M, 48 experts, N=1280=128*10
    ]

    try:
        compiled_quant = torch.compile(mxfp8_quantize_output, fullgraph=True)
        compile_mode = 'fullgraph=True'
    except Exception:
        compiled_quant = torch.compile(mxfp8_quantize_output, fullgraph=False)
        compile_mode = 'fullgraph=False'

    _dummy = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)
    for _ in range(3):
        compiled_quant(_dummy, block_size=32)
    torch.cuda.synchronize()

    print(f'\nArch: SM{arch}0, torch.compile quantize: {compile_mode}')
    print(f'{"G":>3} {"M":>6} {"N":>6} {"K":>6} | {"Fused":>7} | {"BL gem":>7} {"BL qnt":>7} {"BL tot":>7} | {"Spdup":>5} | {"Diff":>8}')
    print('-'*95)

    for num_groups, expected_m, n, k in grouped_shapes:
        m, a, b, grouped_layout, d_dummy, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=use_ue8m0, quant_config=quant_config)

        # Fused
        d_fp8 = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
        d_sf = torch.empty((m, n // 32), device='cuda', dtype=torch.float8_e4m3fn)
        t_fused = bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous_mxfp8out(
            a, b, d_fp8, d_sf, grouped_layout, disable_ue8m0_cast=disable_cast))

        # Baseline gemm
        d_bf16 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        t_bl_gemm = bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d_bf16, grouped_layout, disable_ue8m0_cast=disable_cast))

        # Baseline quantize (compiled)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d_bf16, grouped_layout, disable_ue8m0_cast=disable_cast)
        torch.cuda.synchronize()
        for _ in range(3):
            compiled_quant(d_bf16, block_size=32)
        torch.cuda.synchronize()
        t_bl_quant = bench(lambda: compiled_quant(d_bf16, block_size=32))

        # Accuracy
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous_mxfp8out(a, b, d_fp8, d_sf, grouped_layout, disable_ue8m0_cast=disable_cast)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d_bf16, grouped_layout, disable_ue8m0_cast=disable_cast)
        bl_fp8, bl_sf = mxfp8_quantize_output(d_bf16, block_size=32)
        deq_fused = dequantize_mxfp8(d_fp8, d_sf.view(torch.uint8))
        deq_bl = dequantize_mxfp8(bl_fp8, bl_sf)
        diff = calc_diff(deq_fused, deq_bl)

        t_bl_total = t_bl_gemm + t_bl_quant
        speedup = t_bl_total / t_fused if t_fused > 0 else 0

        print(f'{num_groups:3} {m:6} {n:6} {k:6} | {t_fused:5.3f}ms | {t_bl_gemm:5.3f}ms {t_bl_quant:5.3f}ms {t_bl_total:5.3f}ms | {speedup:4.2f}x | {diff:8.5f}')

    print()


if __name__ == '__main__':
    run_normal_gemm_test()
    run_grouped_gemm_test()
    print('All tests done!')
