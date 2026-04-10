import sys
import torch

sys.path.insert(0, '/mair/team-sys/jangwoong/DeepGEMM/tests')

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import mxfp8_quantize_output

from generators import (
    KernelType, QuantConfig, MajorTypeAB, get_ue8m0_usage, generate_m_grouped_contiguous,
)


def main():
    print('MXFP8 Profiling: G=48, M~60K, N=1280, K=4096')

    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant_config = QuantConfig()
    disable_cast = not use_ue8m0

    num_groups, expected_m, n, k = 48, 1280, 1280, 4096

    m, a, b, grouped_layout, d_dummy, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=use_ue8m0, quant_config=quant_config)

    print(f'Actual M={m}, N={n}, K={k}, G={num_groups}')

    # Fused outputs
    d_fp8 = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
    d_sf = torch.empty((m, n // 32), device='cuda', dtype=torch.float8_e4m3fn)

    # Baseline outputs
    d_bf16 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    compiled_quant = torch.compile(mxfp8_quantize_output, fullgraph=True)

    # Warmup both paths (JIT compile + torch.compile)
    print('Warming up...')
    for _ in range(5):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous_mxfp8out(
            a, b, d_fp8, d_sf, grouped_layout, disable_ue8m0_cast=disable_cast)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d_bf16, grouped_layout, disable_ue8m0_cast=disable_cast)
        compiled_quant(d_bf16, block_size=32)
    torch.cuda.synchronize()

    trace_dir = '/mair/team-sys/jangwoong/DeepGEMM/traces'
    print(f'Profiling to {trace_dir}/')

    # Profile: 10 iterations each, interleaved
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        # 10x fused
        for i in range(10):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous_mxfp8out(
                a, b, d_fp8, d_sf, grouped_layout, disable_ue8m0_cast=disable_cast)

        # 10x baseline (gemm + compiled quantize)
        for i in range(10):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a, b, d_bf16, grouped_layout, disable_ue8m0_cast=disable_cast)
            compiled_quant(d_bf16, block_size=32)

        torch.cuda.synchronize()
        prof.step()

    print(f'Trace saved to {trace_dir}/')
    print('Done!')


if __name__ == '__main__':
    main()
