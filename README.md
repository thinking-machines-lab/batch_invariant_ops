# Batch Invariant Ops

A companion library to the post **“Defeating Nondeterminism in LLM Inference.”** It provides “batch-invariant” replacements for select PyTorch ops so a model’s output for a given element does **not** depend on the batch size it was computed with. This enables truly reproducible inference in settings where batch size varies (e.g., shared servers). [Blog post](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## Why batch invariance?

In typical inference stacks, outputs can change when batch size changes due to floating-point non-associativity and kernel implementation details. Even if per-run kernels are deterministic, *user-visible* results vary when batch composition changes. Batch-invariant kernels remove that dependency by ensuring each element’s result is independent of other elements in the batch.


## Installation

```bash
pip install -e.
```

## Quick Start

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode

# Enable batch-invariant mode
with set_batch_invariant_mode():
    # Your inference code here
    model = YourModel()
    output = model(input_tensor)
```

## Testing Batch-Invariance

The following example shows how batch size can affect results in standard PyTorch:

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

```

## Deterministic Inference in vLLM
`deterministic_vllm_inference.py` shows an proof of concept of validating that vLLM can be made deterministic with a minor upstream PR to use this library. Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream PR, there is only one unique sample.

## Supported Operations

### Matrix Operations
- `torch.mm()` - Matrix multiplication
- `torch.addmm()` - Matrix multiplication with bias addition

### Activation Functions
- `torch.log_softmax()` - Log-softmax activation

### Reduction Operations
- `torch.mean()` - Mean computation along specified dimensions
