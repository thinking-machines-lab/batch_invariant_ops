import torch
from batch_invariant_ops import set_batch_invariant_mode
device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance(dtype=torch.float32):
    M = 32
    K = 128
    N = 1024
    a = torch.linspace(-100, 100, M*K, dtype=dtype).reshape(M, K)

    # Create non-contiguous tensor to mimic the nn.Linear case while weight is always transposed
    # See ref: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50
    b = torch.linspace(-100, 100, K*N, dtype=dtype).reshape(N, K)
    b = b.transpose(0, 1)

    print(f"a is contiguous: {a.is_contiguous()}")
    print(f"b is contiguous: {b.is_contiguous()}")

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
