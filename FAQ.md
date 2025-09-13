# FAQ

### Why do I still see multiple unique completions locally at temperature=0?
Several possibilities:
1. **vLLM isn’t calling the batch-invariant kernels.** The proof-of-concept notes a minor upstream change is required for vLLM to use this library end-to-end. Without it, you can still see multiple unique completions even on a single machine. (See `deterministic_vllm_inference.py` and the blog post.)
2. **Unsupported operations on your model path.** Only a subset of ops are currently batch-invariant (`mm`, `addmm`, `log_softmax`, `mean`). If your model hits other kernels that depend on batch size, results can still differ.
3. **Decode path not strictly deterministic.** Ensure greedy decoding (temperature=0, `top_p=1.0`, `top_k=None`) and fixed seeds for any auxiliary randomness.
4. **Version drift.** Bitwise results can vary across hardware/library versions. Pin PyTorch/CUDA and the inference stack for consistency.

**Checklist to debug**
- Confirm the context manager actually wraps the forward pass (`with set_batch_invariant_mode(True): ...`).
- Start with the provided matrix example; verify `max diff == 0` under batch-invariant mode, then move to your model.
- If using vLLM, follow the proof-of-concept and ensure the upstream hook is active.

---

### What performance overhead should I expect?
It depends on the op and shape; benchmarks are forthcoming. If you have results, please share them via an issue or PR.

### Which ops are planned next?
Open a discussion/issue with your priority ops. We’ll keep a roadmap here once the maintainers set direction.
