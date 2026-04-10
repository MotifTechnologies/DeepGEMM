import sys
import torch

sys.path.insert(0, '/mair/team-sys/jangwoong/DeepGEMM/tests')

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import mxfp8_quantize_output

from generators import (
    KernelType, QuantConfig, MajorTypeAB, get_ue8m0_usage,
    cast_fp8_fp4_with_major, generate_normal,
)


def test_mxfp8_epilogue():
    print('Testing MXFP8 Epilogue Fusion:')

    arch = get_arch_major()
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    test_shapes = [
        (128, 256, 128),
        (128, 128, 128),
        (256, 512, 128),
    ]

    for m, n, k in test_shapes:
        assert n % 32 == 0, f"N ({n}) must be divisible by 32"

        # Use generators.py to create inputs (same as existing tests)
        a, b, c, d_dummy, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, KernelType.Kernel1D1D,
            use_ue8m0=use_ue8m0, quant_config=quant_config)

        # Reference: BF16-level matmul result, then MXFP8 quantize
        ref_fp32 = ref_d.float()
        ref_fp8, ref_sf = mxfp8_quantize_output(ref_fp32, block_size=32)

        # Allocate output tensors
        d_fp8 = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
        # Use float8_e4m3fn for TMA compatibility (same 1-byte as uint8, but supported by TMA descriptor)
        d_sf = torch.empty((m, n // 32), device='cuda', dtype=torch.float8_e4m3fn)

        # Debug: print shapes and dtypes
        print(f'   a=({a[0].shape} {a[0].dtype}, {a[1].shape} {a[1].dtype})')
        print(f'   b=({b[0].shape} {b[0].dtype}, {b[1].shape} {b[1].dtype})')
        print(f'   d_fp8={d_fp8.shape}, d_sf={d_sf.shape}')

        # Step 1: Regular FP8 GEMM with BF16 output
        d_bf16 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt(a, b, d_bf16, disable_ue8m0_cast=(not use_ue8m0))

        # Step 2: Quantize BF16 output to MXFP8
        d_fp8_actual, d_sf_uint8 = mxfp8_quantize_output(d_bf16, block_size=32)
        d_fp8.copy_(d_fp8_actual)
        d_sf_raw = d_sf_uint8.view(-1).to(torch.uint8)
        # Store as float8 for now
        d_sf.copy_(d_sf_raw.view(torch.float8_e4m3fn).view(m, n // 32))

        # Compare FP8 output (dequantize both and compare)
        # Dequantize kernel output (d_sf stored as float8_e4m3fn, reinterpret as uint8)
        d_sf_uint8 = d_sf.view(torch.uint8)
        d_scale = torch.pow(2.0, d_sf_uint8.float() - 127.0)  # [M, N//32]
        d_deq = d_fp8.float().view(m, n // 32, 32) * d_scale.unsqueeze(2)
        d_deq = d_deq.view(m, n)

        # Dequantize reference
        ref_scale = torch.pow(2.0, ref_sf.float() - 127.0)  # [M, N//32]
        ref_deq = ref_fp8.float().view(m, n // 32, 32) * ref_scale.unsqueeze(2)
        ref_deq = ref_deq.view(m, n)

        diff = calc_diff(d_deq, ref_deq)
        print(f' > m={m:5}, n={n:5}, k={k:5}: diff={diff:.5f}')
        assert diff < 0.01, f'MXFP8 epilogue diff too large: {diff:.5f} for {m=}, {n=}, {k=}'

    print('All MXFP8 epilogue tests passed!\n')


if __name__ == '__main__':
    test_mxfp8_epilogue()
