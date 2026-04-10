import sys
import time
import torch

sys.path.insert(0, '/mair/team-sys/jangwoong/DeepGEMM/tests')

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import mxfp8_quantize_output

from generators import (
    KernelType, QuantConfig, MajorTypeAB, get_ue8m0_usage, generate_m_grouped_contiguous,
)


# Register custom op for compiled graph baseline
@torch.library.custom_op("deepgemm::m_grouped_fp8_gemm_nt_bf16out", mutates_args=("d",))
def _m_grouped_fp8_gemm_nt_bf16out(
    a_data: torch.Tensor, a_sf: torch.Tensor,
    b_data: torch.Tensor, b_sf: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    disable_ue8m0_cast: bool,
) -> None:
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_data, a_sf), (b_data, b_sf), d, grouped_layout,
        disable_ue8m0_cast=disable_ue8m0_cast)

@_m_grouped_fp8_gemm_nt_bf16out.register_fake
def _fake(a_data, a_sf, b_data, b_sf, d, grouped_layout, disable_ue8m0_cast):
    pass


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

    # Baseline: compiled graph (GEMM + quantize in one fx graph)
    d_bf16 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    def baseline_fn():
        _m_grouped_fp8_gemm_nt_bf16out(a[0], a[1], b[0], b[1], d_bf16, grouped_layout, disable_cast)
        return mxfp8_quantize_output(d_bf16, block_size=32)

    compiled_bl = torch.compile(baseline_fn, fullgraph=True)

    # Warmup both paths
    print('Warming up...')
    for _ in range(5):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous_mxfp8out(
            a, b, d_fp8, d_sf, grouped_layout, disable_ue8m0_cast=disable_cast)
        compiled_bl()
    torch.cuda.synchronize()

    ts = int(time.time())
    trace_dir = f'/mair/team-sys/jangwoong/DeepGEMM/traces/tma_compiled_{ts}'
    print(f'Profiling to {trace_dir}/')

    # Profile: 10 iterations each
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

        # 10x baseline (compiled graph: gemm + quantize)
        for i in range(10):
            compiled_bl()

        torch.cuda.synchronize()
        prof.step()

    print(f'Trace saved to {trace_dir}/')
    print('Done!')


if __name__ == '__main__':
    main()
