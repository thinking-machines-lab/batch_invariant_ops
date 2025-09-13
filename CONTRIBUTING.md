# Contributing

Thanks for considering a contribution!

## Ways to help
- **Docs:** improve README, add FAQs/tutorials, clarify version support, add examples/benchmarks.
- **Kernels:** propose/implement additional batch-invariant ops.
- **Testing:** add minimal repros and coverage around new ops.

## Pull request process
1. Fork the repo and create a topic branch: `git checkout -b docs/<short-topic>`.
2. Keep PRs small and focused (e.g., “Add FAQ on vLLM determinism + supported ops table”).
3. Include a short rationale and, for code, a minimal test or repro snippet.
4. Be ready to adjust wording/style to match maintainer preferences.

## Style for docs
- Audience: PyTorch/vLLM users; keep examples runnable.
- Prefer small, self-contained code blocks.
- State assumptions and limitations plainly.
- Avoid over-promising; link to the blog post for background.

## Development (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# pytest  # if tests are added later
