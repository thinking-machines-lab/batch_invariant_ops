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


# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    run_iters()
# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    run_iters()
