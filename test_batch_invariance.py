import torch
from batch_invariant_ops import set_batch_invariant_mode
device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance(dtype=torch.float32):
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D, dtype=dtype).reshape(B, D)
    b = torch.linspace(-100, 100, D*D, dtype=dtype).reshape(D, D)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    return diff.item() == 0, diff

def run_iters(iters=10):
    for dtype in [ torch.float32 , torch.bfloat16 ]:
        is_deterministic = True
        difflist = []
        for i in range (iters):
            isd, df = test_batch_invariance(dtype)
            is_deterministic = is_deterministic and isd
            difflist.append(df)
        print( f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations")


def test_fp32_matmul_precision():
    """fp32 matmul must keep IEEE precision, not silently fall back to TF32.

    tl.dot defaults to TF32 for fp32 inputs (10-bit mantissa; integers exact only to
    2**11 = 2048), whereas torch.mm uses IEEE fp32 by default. Without
    input_precision="ieee" in matmul_kernel_persistent, every fp32 mm/addmm under
    batch-invariant mode loses precision for values > 2048 -- e.g. it corrupts the
    rotary-embedding phase (positions @ inv_freq) for long context. Returns the count
    of mismatched elements against the exact integer reference (0 == correct).
    """
    pos  = torch.arange(8192, dtype=torch.float32)   # every value exact in fp32
    ones = torch.ones(1, 1, dtype=torch.float32)
    with set_batch_invariant_mode(True):
        out = torch.mm(ones, pos[None, :])
    mismatches = int((out[0] != pos).sum())
    assert mismatches == 0, (
        f"fp32 matmul lost precision: {mismatches} elements wrong, "
        f"e.g. position 2049 -> {out[0, 2049].item()} (expected 2049.0). "
        f"matmul_kernel_persistent is using TF32 for fp32 inputs."
    )
    return mismatches


# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    run_iters()
# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    run_iters()

# fp32 precision: matmul must not silently downgrade to TF32 (see issue #23)
print("\nfp32 precision (TF32 regression):")
print(f"  fp32 matmul mismatches vs IEEE reference: {test_fp32_matmul_precision()} (expected 0)")
